async function postJson(url, body){
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  return res.json();
}

function el(id){return document.getElementById(id)}

el('predictBtn').addEventListener('click', async () => {
  const dish = el('dish').value.trim();
  const ingredients = el('ingredients').value.trim();
  const payload = { dish_name: dish, ingredients: ingredients };
  el('result').innerHTML = '<div class="small">Predicting…</div>';
  try{
    const data = await postJson('/api/predict', payload);
    if (data.error){
      el('result').innerHTML = `<div class="label">Error</div><div class="value">${data.error}</div>`;
    } else {
      el('result').innerHTML = `<div class="label">Predicted Country</div><div class="value">${data.prediction}</div>`;
    }
  } catch(err){
    el('result').innerHTML = `<div class="label">Error</div><div class="value">${err.message}</div>`;
  }
});

el('lookupBtn').addEventListener('click', async () => {
  const q = el('dish').value.trim();
  if(!q){ alert('Enter a dish name to lookup'); return }
  el('result').innerHTML = '<div class="small">Looking up dish…</div>';
  try{
    const res = await fetch(`/api/lookup?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    if(data.results && data.results.length){
      const first = data.results[0];
      el('ingredients').value = first.ingredients || '';
      el('result').innerHTML = `<div class="label">Found: ${first.dish_name}</div><div class="small">Ingredients loaded into textarea.</div>`;
    } else if (data.error){
      el('result').innerHTML = `<div class="label">Error</div><div class="value">${data.error}</div>`;
    } else {
      el('result').innerHTML = `<div class="label">No results found</div>`;
    }
  }catch(err){
    el('result').innerHTML = `<div class="label">Error</div><div class="value">${err.message}</div>`;
  }
});
