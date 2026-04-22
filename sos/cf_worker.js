export default {
  async fetch(request) {
    const url = new URL(request.url);
    return fetch('https://api.telegram.org' + url.pathname + url.search, {
      method:  request.method,
      headers: request.headers,
      body:    request.body,
    });
  }
}
