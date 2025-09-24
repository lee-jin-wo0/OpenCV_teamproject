// import React, { useState } from 'react'

// export default function UploadDropzone({ onFiles }) {
//   const [drag, setDrag] = useState(false)
//   return (
//     <div className="dropzone"
//       onDragOver={e => {e.preventDefault(); setDrag(true)}}
//       onDragLeave={() => setDrag(false)}
//       onDrop={e => {
//         e.preventDefault(); setDrag(false)
//         const files = Array.from(e.dataTransfer.files)
//         onFiles(files)
//       }}>
//       <p style={{opacity:.8}}>이미지 파일을 드래그&드롭하거나</p>
//       <input id="file" type="file" accept="image/*" multiple
//              style={{display:'none'}}
//              onChange={e => onFiles(Array.from(e.target.files))}/>
//       <label htmlFor="file" className="btn" style={{display:'inline-block', marginTop:12}}>
//         파일 선택
//       </label>
//       {drag && <p style={{marginTop:8}} className="badge">놓으면 업로드됩니다</p>}
//     </div>
//   )
// }
import React, { useState } from 'react'

export default function UploadDropzone({ onFiles }) {
  const [drag, setDrag] = useState(false)

  const handleFiles = (fileList) => {
    const files = Array.from(fileList || [])
    onFiles(files)
  }

  return (
    <div
      className={`dropzone ${drag ? 'drag-active' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault(); setDrag(false)
        handleFiles(e.dataTransfer.files)
      }}
    >
      {/* Simple upload icon */}
      <svg width="48" height="48" viewBox="0 0 24 24" aria-hidden="true"
           style={{ opacity:.9, marginBottom:8 }}>
        <path fill="currentColor" d="M12 3l4 4h-3v6h-2V7H8l4-4zm-7 14v2h14v-2H5z"/>
      </svg>

      <div className="drop-title">Drag & Drop images here</div>
      <div className="drop-sub">또는 아래 버튼으로 파일을 선택하세요</div>

      <input
        id="file"
        type="file"
        accept="image/*"
        multiple
        style={{ display:'none' }}
        onChange={(e) => handleFiles(e.target.files)}
      />

      <label htmlFor="file" className="btn" style={{ marginTop:14 }}>
        파일 선택
      </label>

      {drag && <p className="badge" style={{ marginTop:10 }}>놓으면 업로드됩니다</p>}
    </div>
  )
}