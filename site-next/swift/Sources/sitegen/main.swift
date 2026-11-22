import Foundation
import Evidence

// Sources/sitegen/main.swift -> site-next/swift
let here = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
let siteRoot = here.deletingLastPathComponent()          // site-next
let repoRoot = siteRoot.deletingLastPathComponent()      // raiz do repositório
let webRoot = siteRoot.appendingPathComponent("web")

let started = Date()
let ev = try Evidence(artifactsRoot: repoRoot.appendingPathComponent("artifacts"))
let css = try Design.css(webRoot: webRoot)

let nav = """
<a href="index.html">Home</a><a href="honesty.html">The argument</a><a href="proof.html">Proof</a>
"""

let out = here.appendingPathComponent("dist")
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

let pages: [(String, String, String)] = [
    ("index.html", "Sounio — compute at the boundary between certainty and consequence",
     try Pages.home(ev)),
    ("honesty.html", "Sounio — the argument, in one page", try Pages.honesty(ev)),
    ("proof.html", "Sounio — the corpus", try Pages.proof(ev)),
]

for (file, title, body) in pages {
    let html = Design.page(title: title, css: css, nav: nav, body: body)
    try html.write(to: out.appendingPathComponent(file), atomically: true, encoding: .utf8)
}

// os assets de marca vêm do mesmo lugar que o app React usa
let brandSrc = webRoot.appendingPathComponent("public/brand")
let brandDst = out.appendingPathComponent("brand")
try? FileManager.default.removeItem(at: brandDst)
try? FileManager.default.copyItem(at: brandSrc, to: brandDst)

let s = ev.stats
let ms = Int(Date().timeIntervalSince(started) * 1000)
FileHandle.standardError.write("""
sitegen: \(s.artifacts) artefatos (\(s.gates) portões, \(s.records) registros, \(s.refused) recusas)
         \(s.metrics) métricas · \(pages.count) páginas · \(ms) ms
         → \(out.path)

""".data(using: .utf8)!)
