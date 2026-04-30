async function analyze() {
  const resume = document.getElementById('resume').value;
  const jd = document.getElementById('jd').value;
  const response = await fetch('http://127.0.0.1:8000/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({resume_text: resume, job_description: jd})
  });
  document.getElementById('result').textContent = JSON.stringify(await response.json(), null, 2);
}
