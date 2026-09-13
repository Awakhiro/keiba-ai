"""
同じWi-Fi内のスマホからPCの予想ページを見るための簡易サーバ。

    python serve.py
表示されたURLをスマホのブラウザで開く。ホーム画面に追加すればアプリのように使える。
"""
import http.server, socket, socketserver, os

PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

with socketserver.TCPServer(("0.0.0.0", PORT), H) as httpd:
    print(f"スマホのブラウザで開く:  http://{local_ip()}:{PORT}/予想.html")
    print("停止は Ctrl+C")
    httpd.serve_forever()
