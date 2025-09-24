export async function postProcess(files) {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  const res = await fetch('/api/process', { method: 'POST', body: form })
  if (!res.ok) throw new Error('서버 오류')
  return await res.json()
}