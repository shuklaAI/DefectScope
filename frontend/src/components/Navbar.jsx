import React from "react";
import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Home", exact: true },
  { to: "/inspection", label: "Inspection" },
  { to: "/pipeline", label: "Pipeline" },
  { to: "/results", label: "Results" },
  { to: "/history", label: "History" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/about", label: "Methodology" },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-40 border-b border-ink-600 bg-ink-950">
      <div className="max-w-7xl mx-auto px-5 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-accent-500/15 border border-accent-500/40 flex items-center justify-center">
            <span className="w-2.5 h-2.5 rounded-sm bg-accent-400" />
          </div>
          <div>
            <div className="font-semibold tracking-tight leading-none">DefectScope</div>
            <div className="text-[10px] font-mono text-slate-500 leading-none mt-0.5">DIP INSPECTION SYSTEM</div>
          </div>
        </div>
        <nav className="hidden md:flex items-center gap-1">
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.exact}
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-ink-700 text-accent-400"
                    : "text-slate-400 hover:text-slate-100 hover:bg-ink-800"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
        <NavLink
          to="/inspection"
          className="hidden md:inline-flex items-center gap-2 px-3.5 py-2 rounded-md bg-accent-500 hover:bg-accent-600 text-white text-sm font-semibold transition-colors"
        >
          Start Inspection
        </NavLink>
      </div>
      <nav className="md:hidden flex overflow-x-auto gap-1 px-3 pb-2">
        {LINKS.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.exact}
            className={({ isActive }) =>
              `whitespace-nowrap px-3 py-1.5 rounded-md text-xs font-medium ${
                isActive ? "bg-ink-700 text-accent-400" : "text-slate-400"
              }`
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
