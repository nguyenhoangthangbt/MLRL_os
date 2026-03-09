import { Link, Outlet, useLocation } from 'react-router-dom'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/datasets', label: 'Datasets' },
  { to: '/builder', label: 'Builder' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/models', label: 'Models' },
]

export default function AppShell() {
  const { pathname } = useLocation()

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 text-gray-300 flex flex-col">
        <div className="px-4 py-5 border-b border-gray-700">
          <h1 className="text-lg font-bold text-white">ML/RL OS</h1>
          <p className="text-xs text-gray-500 mt-0.5">v0.1 — Predictive Intelligence</p>
        </div>
        <nav className="flex-1 py-4 space-y-1">
          {NAV.map(({ to, label }) => {
            const active = pathname === to || (to !== '/' && pathname.startsWith(to))
            return (
              <Link
                key={to}
                to={to}
                className={`block px-4 py-2 text-sm transition-colors ${
                  active
                    ? 'bg-gray-800 text-white border-l-2 border-blue-400'
                    : 'hover:bg-gray-800 hover:text-white'
                }`}
              >
                {label}
              </Link>
            )
          })}
        </nav>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>
    </div>
  )
}
