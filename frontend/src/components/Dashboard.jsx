import React from 'react';
import { useAuth } from '../context/AuthContext';
import { supabase } from '../lib/supabase';
import AnalysisWorkspace from './analysis/AnalysisWorkspace';

export default function Dashboard({ onBackToLanding, openAuthModal }) {
  const { user, profile, signOut } = useAuth();

  const displayName = profile?.full_name || user?.email?.split('@')[0] || 'Researcher';

  return (
    <div className="dashboard-page flex flex-col min-h-screen bg-[#030712]">
      {/* Dashboard Top Header Navigation */}
      <header className="dashboard-nav z-50">
        <div className="logo" onClick={onBackToLanding} style={{ cursor: 'pointer' }}>
          <span className="logo-dot"></span>
          SatQuery <span>AI</span>
        </div>

        <div className="dash-user-section">
          {user ? (
            <>
              <div className="user-badge">
                <span className="user-dot"></span>
                <span className="user-name">{displayName}</span>
              </div>

              <button className="secondary-cta dash-landing-btn" onClick={onBackToLanding}>
                Back
              </button>

              <button className="start-btn dash-logout-btn" onClick={signOut}>
                Log Out
              </button>
            </>
          ) : (
            <>
              <button className="secondary-cta dash-landing-btn" onClick={onBackToLanding}>
                 Back
              </button>

              <button className="start-btn" onClick={() => openAuthModal && openAuthModal('login')}>
                Sign In
              </button>
            </>
          )}
        </div>
      </header>

      {/* Main Integrated Analysis Workspace */}
      <main className="flex-1">
        <AnalysisWorkspace supabase={supabase} user={user} openAuthModal={openAuthModal} />
      </main>
    </div>
  );
}
