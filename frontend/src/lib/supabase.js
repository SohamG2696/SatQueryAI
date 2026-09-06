import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn("Supabase environment variables missing! Check .env file.");
} else {
  console.log("Supabase client initialized for URL:", supabaseUrl);
  console.log("Supabase key loaded:", Boolean(supabaseAnonKey));
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);