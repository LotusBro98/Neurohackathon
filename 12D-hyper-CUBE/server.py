from http.server import BaseHTTPRequestHandler,HTTPServer

class HttpProcessor(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('content-type','text/html')
        self.end_headers()
        self.wfile.write(bytes("hello !", "utf-8"))

serv = HTTPServer(("localhost",88),HttpProcessor)
serv.serve_forever()