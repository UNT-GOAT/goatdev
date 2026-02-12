import React from 'react';

function Login({ onLogin, isLogin, toggleLogin }) {
  const handleSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const username = formData.get('username');
    const password = formData.get('password');
    onLogin(username, password);
  };

  return (
    <div className="form-container">
      <h1>{isLogin ? 'Login' : 'Register'}</h1>
      <form onSubmit={handleSubmit}>
        {!isLogin && (
          <div className="form-group">
            <label>Email:</label>
            <input type="email" name="email" required />
          </div>
        )}
        <div className="form-group">
          <label>Username:</label>
          <input type="text" name="username" required />
        </div>
        <div className="form-group">
          <label>Password:</label>
          <input type="password" name="password" required />
        </div>
        {!isLogin && (
          <div className="form-group">
            <label>Confirm Password:</label>
            <input type="password" name="confirmPassword" required />
          </div>
        )}
        <button type="submit" className="form-button">
          {isLogin ? 'Login' : 'Register'}
        </button>
      </form>
      <p className="toggle-text">
        {isLogin ? "Don't have an account?" : 'Already have an account?'}{' '}
        <span className="toggle-link" onClick={toggleLogin}>
          {isLogin ? 'Register' : 'Login'}
        </span>
      </p>
    </div>
  );
}

export default Login;