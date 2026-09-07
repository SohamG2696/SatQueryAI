import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || "https://placeholder-project.supabase.co";
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.placeholder";

if (!import.meta.env.VITE_SUPABASE_URL || !import.meta.env.VITE_SUPABASE_ANON_KEY) {
  console.warn("Supabase environment variables missing! Running with fallback client.");
} else {
  console.log("Supabase client initialized for URL:", supabaseUrl);
  console.log("Supabase key loaded:", Boolean(supabaseAnonKey));
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);