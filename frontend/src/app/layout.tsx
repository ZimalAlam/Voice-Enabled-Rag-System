import './globals.css'

export const metadata = {
  title: 'Agentic RAG AI Assistant',
  description: 'AI assistant with document analysis, web search, and voice input capabilities',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}
