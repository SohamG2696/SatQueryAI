import React, { createContext, useContext, useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch user profile from public.profiles or user_metadata fallback
  const fetchProfile = async (userId) => {
    if (!userId) {
      setProfile(null);
      return;
    }
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single();

      if (!error && data) {
        setProfile(data);
      } else {
        // Fallback to user metadata if profile table query is restricted or building
        const { data: userData } = await supabase.auth.getUser();
        const metaName = userData?.user?.user_metadata?.full_name;
        if (metaName) {
          setProfile({ id: userId, full_name: metaName });
        } else {
          setProfile(null);
        }
      }
    } catch (err) {
      console.warn('Profile fetch warning:', err.message);
    }
  };

  useEffect(() => {
    // 1. Check existing session on mount
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.user) {
        fetchProfile(session.user.id);
      }
      setLoading(false);
    }).catch(err => {
      console.warn('Session retrieval check:', err.message);
      setLoading(false);
    });

    // 2. Listen for auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(async (event, session) => {
      setSession(session);
      setUser(session?.user ?? null);

      if (session?.user) {
        await fetchProfile(session.user.id);
      } else {
        setProfile(null);
      }

      setLoading(false);
    });

    return () => {
      subscription?.unsubscribe();
    };
  }, []);

  // Sign In with email & password
  const signIn = async ({ email, password }) => {
    console.log("Attempting Supabase auth.signInWithPassword for:", email);
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) {
      console.error("AUTH SIGNIN ERROR:", error);
      throw error;
    }
    console.log("AUTH SIGNIN SUCCESS:", data?.user);
    return data;
  };

  // Sign Up passing full_name in options.data AND fallback upsert
  const signUp = async ({ email, password, fullName }) => {
    console.log("Attempting Supabase auth.signUp with metadata for:", email);
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName,
        },
      },
    });

    if (error) {
      console.error("AUTH SIGNUP ERROR:", error);
      throw error;
    }

    console.log("AUTH SIGNUP SUCCESS (user metadata passed):", data?.user);

    // Safeguard: Also perform direct client upsert into public.profiles if user ID exists
    if (data?.user) {
      try {
        const { error: profileErr } = await supabase.from('profiles').upsert(
          {
            id: data.user.id,
            full_name: fullName,
            created_at: new Date().toISOString(),
          },
          { onConflict: 'id' }
        );

        if (profileErr) {
          console.warn("Client-side profile upsert note (database trigger may be responsible):", profileErr.message);
        } else {
          console.log("Client-side profile upsert SUCCESS for user:", data.user.id);
        }
      } catch (pErr) {
        console.warn("Client-side profile upsert exception:", pErr.message);
      }
    }

    return data;
  };

  // Reset Password
  const resetPassword = async (email) => {
    console.log("Attempting Supabase auth.resetPasswordForEmail for:", email);
    const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/`,
    });
    if (error) {
      console.error("AUTH RESET ERROR:", error);
      throw error;
    }
    return data;
  };

  // Sign Out
  const signOut = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) console.error('Signout error:', error.message);
    setSession(null);
    setUser(null);
    setProfile(null);
  };

  const value = {
    user,
    session,
    profile,
    loading,
    signIn,
    signUp,
    resetPassword,
    signOut,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
