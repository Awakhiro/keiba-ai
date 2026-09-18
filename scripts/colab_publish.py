"""
private リポジトリで集めたデータを Colab に持ってきて、Google Drive に予想ページを書き出す。

重い収集は GitHub Actions（private）に任せ、ページ作りだけ Colab で行う構成。
GitHub Pages を使わないのでリポジトリを公開する必要がなく、
できたページは Drive アプリからタップで開ける。

    from scripts.colab_publish import publish_to_drive
    publish_to_drive(token=TOKEN, owner="yourname", repo="keiba-ai",
                     out_dir="/content/drive/MyDrive/keiba_ai")
"""

import os
import re
import shutil
import subprocess


def _run(args, secret=None, **kw):
    """
    シェルを介さずにコマンドを実行する。
    エラー文にトークンが混じらないよう、secret は伏せ字に置き換える。
    """
    r = subprocess.run(args, shell=False, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "")[-800:]
        shown = " ".join(args)
        if secret:
            msg = msg.replace(secret, "***")
            shown = shown.replace(secret, "***")
        raise RuntimeError(f"失敗: {shown}\n{msg}")
    return r.stdout


def _clean_token(token):
    """
    貼り付けのゆらぎを吸収する。
    トークンに空白は含まれないので、空白や改行は安全に取り除ける。
    取り除いたうえで形式が合わなければ、そのときに弾く。
    """
    raw = token or ""
    t = re.sub(r"\s+", "", raw)
    if not t:
        raise ValueError("トークンが空です。")
    if t != raw.strip():
        print("トークンから空白・改行を取り除きました。")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{20,255}", t):
        raise ValueError(
            f"トークンの形式が正しくありません（読み取った長さ {len(t)}）。"
            "前の出力ごとコピーしていないか確認してください。"
            "作り直す場合は GitHub の Settings → Developer settings → "
            "Personal access tokens (classic) で、スコープに repo と workflow を付けて生成し、"
            "表示直後のコピーボタンを使ってください。")
    return t


def _clean_name(name, label):
    n = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9\-_.]{0,98})", n):
        raise ValueError(f"{label} が正しくありません: {n!r}")
    return n


def clone_private(token, owner, repo, dest="/content/repo"):
    """private リポジトリを取得する（履歴なしの浅いクローン）。"""
    token = _clean_token(token)
    owner = _clean_name(owner, "ユーザー名")
    repo = _clean_name(repo, "リポジトリ名")

    if os.path.exists(dest):
        shutil.rmtree(dest)
    url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
    try:
        _run(["git", "clone", "--depth", "1", url, dest], secret=token)
    except RuntimeError as e:
        if "Authentication failed" in str(e) or "not found" in str(e).lower():
            raise RuntimeError(
                f"{owner}/{repo} を取得できませんでした。トークンの権限（repo）と、"
                "ユーザー名・リポジトリ名の綴りを確認してください。") from None
        raise
    # 認証情報が残らないように接続先を書き換えておく
    _run(["git", "-C", dest, "remote", "set-url", "origin",
          f"https://github.com/{owner}/{repo}.git"])
    return dest


def publish_to_drive(token, owner, repo, out_dir, test_start="2026-08-01",
                     min_grade="B", dest="/content/repo",
                     code_dir="/content/keiba_ai"):
    """
    private リポジトリの「データ」と、ノートブックが持つ「最新のコード」を組み合わせて
    学習し、Drive に 予想.html と 検証レポート.md を置く。

    リポジトリ側のコードが古くても、ノートブックを更新すれば最新の処理で動く。
    戻り値は書き出したHTMLのパス。
    """
    path = clone_private(token, owner, repo, dest)

    src_data = os.path.join(path, "data")
    data_files = os.listdir(src_data) if os.path.isdir(src_data) else []
    if not any(f.startswith("races.") for f in data_files):
        raise FileNotFoundError(
            "リポジトリにまだデータがありません。収集ワークフローの完了を待ってください。")

    # コードは code_dir のものを使う（無ければクローンしたものを使う）
    work = code_dir if os.path.isdir(os.path.join(code_dir, "scripts")) else path
    if work != path:
        dst_data = os.path.join(work, "data")
        os.makedirs(dst_data, exist_ok=True)
        for f in data_files:
            shutil.copy(os.path.join(src_data, f), os.path.join(dst_data, f))
        print(f"データ {len(data_files)}件をコピーし、最新コードで処理します")

    import sys
    r = subprocess.run(
        [sys.executable, "-u", "scripts/train_and_publish.py",
         "--test-start", test_start, "--min-grade", min_grade],
        cwd=work, capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])

    os.makedirs(out_dir, exist_ok=True)
    html = os.path.join(out_dir, "予想.html")
    made = os.path.join(work, "docs", "index.html")
    if not os.path.exists(made):
        raise RuntimeError("ページが作られませんでした。上のログを確認してください。")
    shutil.copy(made, html)
    report = os.path.join(work, "docs", "report.md")
    if os.path.exists(report):
        shutil.copy(report, os.path.join(out_dir, "検証レポート.md"))

    size = os.path.getsize(html) / 1024
    print(f"\n書き出し: {html}  ({size:.0f} KB)")
    print("Drive アプリの keiba_ai フォルダから開けます。")
    return html
