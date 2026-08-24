// Inline SVG icons, stroke-based so they inherit currentColor and the button's text size.
const S = ({ children, size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: '0 0 auto' }}>{children}</svg>
)

export const Send = (p) => <S {...p}><path d="M22 2 11 13" /><path d="M22 2 15 22l-4-9-9-4 20-7z" /></S>
export const Stop = (p) => <S {...p}><rect x="6" y="6" width="12" height="12" rx="2" /></S>
export const Trash = (p) => <S {...p}><path d="M3 6h18" /><path d="M8 6V4h8v2" /><path d="M19 6l-1 14H6L5 6" /></S>
export const Play = (p) => <S {...p}><path d="m6 4 14 8-14 8V4z" /></S>
export const Download = (p) => <S {...p}><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M4 21h16" /></S>
export const Plus = (p) => <S {...p}><path d="M12 5v14" /><path d="M5 12h14" /></S>
export const Save = (p) => <S {...p}><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><path d="M17 21v-8H7v8" /><path d="M7 3v5h8" /></S>
export const Key = (p) => <S {...p}><circle cx="7.5" cy="15.5" r="4.5" /><path d="m10.7 12.3 8.3-8.3" /><path d="m17 6 3 3" /></S>
export const Refresh = (p) => <S {...p}><path d="M21 12a9 9 0 1 1-3-6.7" /><path d="M21 3v6h-6" /></S>
export const Logout = (p) => <S {...p}><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><path d="m16 17 5-5-5-5" /><path d="M21 12H9" /></S>
export const Bolt = (p) => <S {...p}><path d="M13 2 3 14h8l-1 8 10-12h-8l1-8z" /></S>
