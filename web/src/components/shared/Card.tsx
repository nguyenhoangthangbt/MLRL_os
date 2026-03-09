import type { ReactNode } from 'react'

interface Props {
  title?: string
  children: ReactNode
  className?: string
}

export default function Card({ title, children, className = '' }: Props) {
  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {title && (
        <div className="px-5 py-3 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
        </div>
      )}
      <div className="px-5 py-4">{children}</div>
    </div>
  )
}
