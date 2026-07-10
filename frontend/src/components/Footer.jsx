const TEAM = [
  'Saptak Chakraborty',
  'Eric Nicolas Schaubs',
  'Elaheh Khoddam',
  'Sebastian Hauer',
]

export default function Footer() {
  return (
    <footer className="app-footer">
      <p>
        🎬 CineProfile — built by {TEAM.join(' · ')}
      </p>
    </footer>
  )
}
