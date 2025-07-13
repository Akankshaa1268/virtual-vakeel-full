import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="nav-left">
        <Link to="/" className="nav-logo">Virtual Vakil</Link>
      </div>
      <div className="nav-right">
        <Link to="/">Home</Link>
        <Link to="/glossary">Glossary</Link>
        <Link to="/tracker">Tracker</Link>
        <Link to="/templates">Templates</Link>
        <Link to="/login">Login</Link>
      </div>
      
    </nav>
  );
};

export default Navbar;
