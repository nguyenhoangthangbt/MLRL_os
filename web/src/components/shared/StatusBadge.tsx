const COLORS: Record<string, string> = {
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  pending: 'bg-yellow-100 text-yellow-800',
  running: 'bg-blue-100 text-blue-800',
}

export default function StatusBadge({ status }: { status: string }) {
  const color = COLORS[status] ?? 'bg-gray-100 text-gray-800'
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded-full ${color}`}>
      {status}
    </span>
  )
}
