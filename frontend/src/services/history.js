/**
 * SatQuery AI — Persistent Query & Image History Service
 * Handles database operations and Supabase Storage interactions for query history.
 */

/**
 * Saves a completed analysis query and its associated image assets to Supabase.
 */
export async function saveQueryHistory({ supabase, user, query, images = [], response, analysisResult }) {
  if (!supabase || !user?.id) {
    console.warn("Save history skipped: User unauthenticated or Supabase missing.");
    return null;
  }

  try {
    const historyPayload = {
      user_id: user.id,
      query: query || "Satellite Scene Inspection",
      task_detected: response?.task_detected || analysisResult?.task_detected || "vqa",
      model_used: analysisResult?.model || response?.execution_summary?.models_used?.[0] || "SatQuery Controller",
      confidence: analysisResult?.confidence || (response?.confidence != null ? `${Math.round(response.confidence * 100)}%` : null),
      answer: response?.answer || analysisResult?.message || "",
      execution_summary: {
        ...(response?.execution_summary || {}),
        raw_result: analysisResult || null,
        raw_response: response || null,
      },
      visual_evidence: response?.visual_evidence || analysisResult?.visual_evidence || null,
      created_at: new Date().toISOString(),
    };

    // 1. Insert into public.query_history
    const { data: historyData, error: historyErr } = await supabase
      .from("query_history")
      .insert(historyPayload)
      .select()
      .single();

    if (historyErr) {
      console.warn("query_history table insert error:", historyErr.message);
      // Fallback: also update/insert existing 'analyses' table for backwards compatibility
      await supabase.from("analyses").insert({
        user_id: user.id,
        query: historyPayload.query,
        ai_response: historyPayload.answer,
        created_at: historyPayload.created_at,
      }).catch(() => {});
      return null;
    }

    const historyId = historyData.id;

    // 2. Upload images to Supabase Storage 'history-images' bucket
    const imageRecords = [];
    for (let i = 0; i < images.length; i++) {
      const imgObj = images[i];
      const rawFile = imgObj.file || imgObj;

      if (rawFile instanceof File || rawFile instanceof Blob) {
        const fileName = imgObj.name || rawFile.name || `image_${i + 1}.png`;
        const sanitizedFileName = fileName.replace(/[^a-zA-Z0-9._-]/g, "_");
        const storagePath = `${user.id}/${historyId}/${Date.now()}_${i}_${sanitizedFileName}`;

        try {
          const { error: uploadErr } = await supabase.storage
            .from("history-images")
            .upload(storagePath, rawFile, { upsert: true });

          if (uploadErr) {
            console.warn(`Storage upload error for image ${fileName}:`, uploadErr.message);
            continue;
          }

          const { data: urlData } = supabase.storage
            .from("history-images")
            .getPublicUrl(storagePath);

          const publicUrl = urlData?.publicUrl || null;

          imageRecords.push({
            history_id: historyId,
            user_id: user.id,
            storage_path: storagePath,
            file_name: fileName,
            file_size: imgObj.size || `${(rawFile.size / (1024 * 1024)).toFixed(2)} MB`,
            file_type: rawFile.type || "image/png",
            public_url: publicUrl,
            image_index: i,
            created_at: new Date().toISOString(),
          });
        } catch (imgErr) {
          console.warn("Image upload exception:", imgErr.message);
        }
      }
    }

    // 3. Insert image metadata records
    if (imageRecords.length > 0) {
      const { error: imgRecordsErr } = await supabase
        .from("query_history_images")
        .insert(imageRecords);

      if (imgRecordsErr) {
        console.warn("query_history_images insert note (retrying without image_index):", imgRecordsErr.message);
        // Fallback without image_index column if table was created without it
        const fallbackRecords = imageRecords.map(({ image_index, ...rest }) => rest);
        await supabase.from("query_history_images").insert(fallbackRecords).catch(() => {});
      }
    }

    return historyData;
  } catch (err) {
    console.error("Save query history failed:", err);
    return null;
  }
}

/**
 * Fetches persistent query history for the logged-in user.
 */
export async function getUserQueryHistory({ supabase, user }) {
  if (!supabase || !user?.id) return [];

  try {
    const { data: historyList, error: historyErr } = await supabase
      .from("query_history")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false });

    if (historyErr) {
      console.warn("Error fetching query_history:", historyErr.message);
      // Fallback: try reading from legacy 'analyses' table
      const { data: legacyData } = await supabase
        .from("analyses")
        .select("*")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false });

      if (legacyData) {
        return legacyData.map((item) => ({
          id: item.id,
          query: item.query,
          answer: item.ai_response,
          task_detected: "vqa",
          model_used: "SatQuery Controller",
          confidence: "Validated",
          created_at: item.created_at,
          images: [],
        }));
      }
      return [];
    }

    if (!historyList || historyList.length === 0) return [];

    // Fetch images associated with these history entries
    const historyIds = historyList.map((item) => item.id);
    const { data: imageList } = await supabase
      .from("query_history_images")
      .select("*")
      .in("history_id", historyIds);

    const imagesByHistoryId = {};
    if (imageList) {
      imageList.forEach((img) => {
        if (!imagesByHistoryId[img.history_id]) {
          imagesByHistoryId[img.history_id] = [];
        }
        imagesByHistoryId[img.history_id].push(img);
      });

      // Sort images for each history item by image_index or created_at
      Object.keys(imagesByHistoryId).forEach((hId) => {
        imagesByHistoryId[hId].sort((a, b) => {
          if (a.image_index != null && b.image_index != null) {
            return a.image_index - b.image_index;
          }
          return new Date(a.created_at || 0) - new Date(b.created_at || 0);
        });
      });
    }

    return historyList.map((item) => ({
      ...item,
      images: imagesByHistoryId[item.id] || [],
    }));
  } catch (err) {
    console.error("getUserQueryHistory failed:", err);
    return [];
  }
}

/**
 * Deletes a query history item and its associated storage assets.
 */
export async function deleteQueryHistoryItem({ supabase, user, historyId }) {
  if (!supabase || !user?.id || !historyId) return false;

  try {
    // 1. Fetch images to retrieve storage paths
    const { data: images } = await supabase
      .from("query_history_images")
      .select("storage_path")
      .eq("history_id", historyId)
      .eq("user_id", user.id);

    if (images && images.length > 0) {
      const storagePaths = images.map((img) => img.storage_path);
      await supabase.storage.from("history-images").remove(storagePaths);
    }

    // 2. Delete history record (cascade deletes image metadata)
    const { error } = await supabase
      .from("query_history")
      .delete()
      .eq("id", historyId)
      .eq("user_id", user.id);

    if (error) {
      console.warn("Delete query_history error:", error.message);
      // Fallback try legacy analyses table
      await supabase.from("analyses").delete().eq("id", historyId).catch(() => {});
    }

    return true;
  } catch (err) {
    console.error("deleteQueryHistoryItem failed:", err);
    return false;
  }
}

/**
 * Downloads stored images for a history item and restores them as active File payloads for ImageUploader.
 */
export async function restoreHistoryImages({ supabase, imageRecords = [] }) {
  if (!imageRecords || imageRecords.length === 0) return [];

  const restoredPayloads = [];

  // Sort image records by image_index or created_at
  const sortedRecords = [...imageRecords].sort((a, b) => {
    if (a.image_index != null && b.image_index != null) {
      return a.image_index - b.image_index;
    }
    return new Date(a.created_at || 0) - new Date(b.created_at || 0);
  });

  for (const imgRec of sortedRecords) {
    try {
      let blob = null;

      // Method 1: Try downloading directly via Supabase storage client
      if (supabase && imgRec.storage_path) {
        const { data, error } = await supabase.storage
          .from("history-images")
          .download(imgRec.storage_path);

        if (!error && data) {
          blob = data;
        } else if (error) {
          console.warn(`Supabase storage download note for ${imgRec.storage_path}:`, error.message);
        }
      }

      // Method 2: Try signed URL download via Supabase storage client
      if (!blob && supabase && imgRec.storage_path) {
        try {
          const { data: signedData } = await supabase.storage
            .from("history-images")
            .createSignedUrl(imgRec.storage_path, 3600);

          if (signedData?.signedUrl) {
            const res = await fetch(signedData.signedUrl);
            if (res.ok) {
              blob = await res.blob();
            }
          }
        } catch (signedErr) {
          console.warn("Signed URL download note:", signedErr.message);
        }
      }

      // Method 3: Fallback to public_url fetch
      if (!blob && imgRec.public_url) {
        try {
          const res = await fetch(imgRec.public_url);
          if (res.ok) {
            blob = await res.blob();
          }
        } catch (fetchErr) {
          console.warn("Public URL fetch note:", fetchErr.message);
        }
      }

      if (blob) {
        const fileName = imgRec.file_name || "restored_image.png";
        const fileType = blob.type || imgRec.file_type || "image/png";
        const file = new File([blob], fileName, { type: fileType, lastModified: Date.now() });

        restoredPayloads.push({
          isCustom: true,
          file,
          id: `${file.name}-${file.lastModified}-${file.size}-${Math.random()}`,
          name: file.name,
          ext: file.name.split(".").pop().toUpperCase(),
          size: imgRec.file_size || `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
          url: URL.createObjectURL(file),
          sensor: "Optical / SAR Sensor",
          resolution: "0.5m GSD",
          coordinates: "Auto-georeferenced",
          captureDate: new Date(imgRec.created_at || Date.now()).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
          }),
        });
      } else {
        console.warn(`Could not retrieve image blob for: ${imgRec.file_name}`);
      }
    } catch (err) {
      console.warn(`Failed to restore image ${imgRec.file_name}:`, err.message);
    }
  }

  return restoredPayloads;
}
