export async function postProcess(files) {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  
  // 백엔드 서버의 절대 주소로 수정
  const res = await fetch('http://127.0.0.1:8000/process', { method: 'POST', body: form })
  
  if (!res.ok) throw new Error('서버 오류')
  return await res.json()
}