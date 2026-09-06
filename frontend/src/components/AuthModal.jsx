import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

export default function AuthModal({ isOpen, onClose, initialMode = 'login' }) {
  const { signIn, signUp, resetPassword } = useAuth();

  const [mode, setMode] = useState(initialMode); // 'login' | 'signup' | 'forgot'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  useEffect(() => {
    setMode(initialMode);
    setErrorMsg('');
    setSuccessMsg('');
    setPassword('');
    setConfirmPassword('');
  }, [initialMode, isOpen]);

  if (!isOpen) return null;

  const handleClose = () => {
    setErrorMsg('');
    setSuccessMsg('');
    onClose();
  };

  const validateForm = () => {
    setErrorMsg('');
    setSuccessMsg('');

    if (!email.trim() || !email.includes('@')) {
      setErrorMsg('Please enter a valid email address.');
      return false;
    }

    if (mode === 'signup') {
      if (!fullName.trim()) {
        setErrorMsg('Please enter your full name.');
        return false;
      }
      if (password.length < 6) {
        setErrorMsg('Password must be at least 6 characters long.');
        return false;
      }
      if (password !== confirmPassword) {
        setErrorMsg('Passwords do not match.');
        return false;
      }
    }

    if (mode === 'login' && !password) {
      setErrorMsg('Please enter your password.');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setSubmitting(true);
    setErrorMsg('');
    setSuccessMsg('');

    try {
      if (mode === 'login') {
        await signIn({ email: email.trim(), password });
        handleClose();
      } else if (mode === 'signup') {
        const data = await signUp({
          email: email.trim(),
          password,
          fullName: fullName.trim(),
        });

        if (data?.session) {
          setSuccessMsg('Account created successfully!');
          setTimeout(() => handleClose(), 1200);
        } else {
          setSuccessMsg('Account created! Please check your email to confirm verification.');
        }
      } else if (mode === 'forgot') {
        await resetPassword(email.trim());
        setSuccessMsg('Password reset instructions sent! Please check your inbox.');
      }
    } catch (err) {
      console.error('Auth action error:', err);
      let msg = err.message || 'An error occurred. Please try again.';

      if (msg.includes('Invalid login credentials')) {
        msg = 'Invalid email or password.';
      } else if (msg.includes('User already registered')) {
        msg = 'An account with this email already exists.';
      } else if (msg.includes('Email not confirmed')) {
        msg = 'Please verify your email address before logging in.';
      } else if (msg.toLowerCase().includes('rate limit')) {
        msg = 'Supabase email rate limit reached. Please disable "Confirm email" in Supabase Dashboard (Authentication -> Providers -> Email) to allow instant account creation during testing.';
      }

      setErrorMsg(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-overlay" onClick={handleClose}>
      <div className="auth-panel" onClick={(e) => e.stopPropagation()}>
        {/* Close Icon Button */}
        <button className="auth-close-btn" onClick={handleClose} aria-label="Close modal">
          ✕
        </button>

        {/* Modal Header */}
        <div className="auth-header">
          <div className="auth-logo">
            <span className="logo-dot"></span>
            SatQuery <span>AI</span>
          </div>

          <h2>
            {mode === 'login' && 'Welcome Back'}
            {mode === 'signup' && 'Create Your Account'}
            {mode === 'forgot' && 'Reset Password'}
          </h2>
          <p>
            {mode === 'login' && 'Access vision-language satellite intelligence.'}
            {mode === 'signup' && 'Join SatQuery AI for intelligent Earth observation.'}
            {mode === 'forgot' && 'Enter your registered email to receive a password reset link.'}
          </p>
        </div>

        {/* Feedback Messages */}
        {errorMsg && <div className="auth-alert auth-alert-error">{errorMsg}</div>}
        {successMsg && <div className="auth-alert auth-alert-success">{successMsg}</div>}

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="auth-form">
          {mode === 'signup' && (
            <div className="auth-field">
              <label htmlFor="fullName">Full Name</label>
              <input
                id="fullName"
                type="text"
                placeholder="e.g. Dr. Alex Morgan"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                disabled={submitting}
                required
              />
            </div>
          )}

          <div className="auth-field">
            <label htmlFor="authEmail">Email Address</label>
            <input
              id="authEmail"
              type="email"
              placeholder="name@organization.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={submitting}
              required
            />
          </div>

          {mode !== 'forgot' && (
            <div className="auth-field">
              <div className="auth-field-row">
                <label htmlFor="authPassword">Password</label>
                {mode === 'login' && (
                  <button
                    type="button"
                    className="auth-link-btn"
                    onClick={() => { setMode('forgot'); setErrorMsg(''); setSuccessMsg(''); }}
                  >
                    Forgot password?
                  </button>
                )}
              </div>
              <input
                id="authPassword"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={submitting}
                required
              />
            </div>
          )}

          {mode === 'signup' && (
            <div className="auth-field">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                id="confirmPassword"
                type="password"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                disabled={submitting}
                required
              />
            </div>
          )}

          {/* Submit CTA Button */}
          <button type="submit" className="auth-submit-btn" disabled={submitting}>
            {submitting ? (
              <span className="auth-spinner">Processing...</span>
            ) : (
              <>
                {mode === 'login' && 'Sign In'}
                {mode === 'signup' && 'Create Account'}
                {mode === 'forgot' && 'Send Reset Link'}
              </>
            )}
          </button>
        </form>

        {/* Footer Navigation Tabs */}
        <div className="auth-footer">
          {mode === 'login' && (
            <p>
              Don't have an account?{' '}
              <button
                type="button"
                className="auth-switch-btn"
                onClick={() => { setMode('signup'); setErrorMsg(''); setSuccessMsg(''); }}
              >
                Get Started
              </button>
            </p>
          )}

          {mode === 'signup' && (
            <p>
              Already have an account?{' '}
              <button
                type="button"
                className="auth-switch-btn"
                onClick={() => { setMode('login'); setErrorMsg(''); setSuccessMsg(''); }}
              >
                Log In
              </button>
            </p>
          )}

          {mode === 'forgot' && (
            <p>
              Remembered your password?{' '}
              <button
                type="button"
                className="auth-switch-btn"
                onClick={() => { setMode('login'); setErrorMsg(''); setSuccessMsg(''); }}
              >
                Back to Login
              </button>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
