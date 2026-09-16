"""
Colab から GitHub リポジトリを作り、プロジェクト一式を置く。

スマホだけで GitHub Actions のセットアップを終わらせるためのもの。
ブラウザで何十個もファイルを作る代わりに、API でまとめて送る。

    from scripts.push_to_github import publish
    publish(token="ghp_xxx", repo="keiba-ai", private=False)

トークンは github.com → Settings → Developer settings → Personal access tokens
→ Tokens (classic) で作成し、`repo` と `workflow` にチェックを入れる。
"""

import base64
import os
import time

import requests

API = "https://api.github.com"
SKIP_DIRS = {"__pycache__", ".git", "data", "docs", ".ipynb_checkpoints"}
SKIP_EXT = {".pyc", ".parquet", ".pkl"}
INCLUDE_ROOT = {
    "requirements.txt", "README.md", "build_site.py", "serve.py",
    "run_demo.py", "make_sample_data.py",
}


def _files(root):
    """送るファイルを集める。データや生成物は送らない。"""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            top = rel.split("/")[0]
            if os.path.splitext(fn)[1] in SKIP_EXT:
                continue
            if "/" not in rel and rel not in INCLUDE_ROOT:
                continue
            if top not in INCLUDE_ROOT and top not in {"src", "webapp", "scripts", ".github"}:
                continue
            with open(full, "rb") as f:
                out[rel] = f.read()
    return out


def publish(token, repo, root=".", private=False, branch="main", pause=0.35):
    from .colab_publish import _clean_token, _clean_name
    token = _clean_token(token)
    repo = _clean_name(repo, "リポジトリ名")
    h = {"Authorization": f"Bearer {token}",
         "Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}

    me = requests.get(f"{API}/user", headers=h, timeout=30)
    me.raise_for_status()
    owner = me.json()["login"]

    r = requests.get(f"{API}/repos/{owner}/{repo}", headers=h, timeout=30)
    if r.status_code == 404:
        r = requests.post(f"{API}/user/repos", headers=h, timeout=30, json={
            "name": repo, "private": private, "auto_init": True,
            "description": "中央競馬の予想AI（自動収集・学習・公開）"})
        r.raise_for_status()
        print(f"リポジトリを作成: {owner}/{repo}")
        time.sleep(2)
    else:
        r.raise_for_status()
        print(f"既存のリポジトリを更新: {owner}/{repo}")
    default_branch = r.json().get("default_branch", branch)

    files = _files(root)
    print(f"{len(files)}ファイルを送信します")
    for i, (path, blob) in enumerate(sorted(files.items()), 1):
        url = f"{API}/repos/{owner}/{repo}/contents/{path}"
        cur = requests.get(url, headers=h, params={"ref": default_branch}, timeout=30)
        body = {"message": f"add {path}", "branch": default_branch,
                "content": base64.b64encode(blob).decode()}
        if cur.status_code == 200:
            body["sha"] = cur.json()["sha"]
        put = requests.put(url, headers=h, json=body, timeout=60)
        if put.status_code not in (200, 201):
            print(f"  失敗 {path}: {put.status_code} {put.text[:160]}")
        else:
            print(f"  [{i:>2}/{len(files)}] {path}")
        time.sleep(pause)

    print(f"\n完了: https://github.com/{owner}/{repo}")
    print("次にやること:")
    print(f"  1. https://github.com/{owner}/{repo}/actions/workflows/collect.yml")
    print("     → Run workflow を押して収集を開始（スマホを閉じても進みます）")
    print(f"  2. https://github.com/{owner}/{repo}/settings/pages")
    print(f"     → Source を「Deploy from a branch」、{default_branch} / docs にして保存")
    print(f"  3. 完成した予想ページ: https://{owner}.github.io/{repo}/")
    return f"https://github.com/{owner}/{repo}"
