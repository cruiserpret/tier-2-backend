import { Link, useLocation } from "react-router-dom";

export function TopNav() {
  const loc = useLocation();
  const onHome = loc.pathname === "/dtc-v3" || loc.pathname === "/";
  return (
    <nav className="topnav">
      <Link to="/dtc-v3" className="topnav-brand">
        <span className="topnav-bracket">[</span>
        <span className="display topnav-wordmark">ASSEMBLY</span>
        <span className="topnav-bracket">]</span>
        <span className="topnav-mode mono">MARKET</span>
      </Link>
      <div className="topnav-meta mono">
        {onHome ? "v3-lite" : (
          <Link to="/dtc-v3" className="topnav-back">+ New simulation</Link>
        )}
      </div>
    </nav>
  );
}
