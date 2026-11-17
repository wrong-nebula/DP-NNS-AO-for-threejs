from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import base64
root=Path(__file__).resolve().parents[1]; out=root/'output/playwright/traced-run2'
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(root),**kw)
 def do_POST(self):
  name=self.path.removeprefix('/save/')
  if not self.path.startswith('/save/') or '/' in name or '..' in name:self.send_error(400);return
  data=self.rfile.read(int(self.headers['Content-Length']))
  if name.endswith('.png'):data=base64.b64decode(data.split(b',',1)[1])
  (out/name).write_bytes(data);self.send_response(200);self.end_headers();self.wfile.write(b'ok')
ThreadingHTTPServer(('127.0.0.1',8918),Handler).serve_forever()
