import Foundation

/// Portão declara veredito; registro traz medição e procedência.
public enum Kind: String, Sendable { case gate, record }

/// O que uma amostra finita sustenta, a 95% (Wilson).
public struct Proportion: Sendable {
    public let passed: Int
    public let total: Int
    public let point: Double
    public let lo: Double
    public let hi: Double
    public var width: Double { hi - lo }
}

public struct Artifact: Sendable {
    public let id: String
    public let kind: Kind
    /// Presente só quando o artefato reporta passed e total.
    public let proportion: Proportion?
    public let territory: String
    public let status: String?
    /// Null em registros: não são portões, não têm veredito.
    public let level: Level?
    public let reason: String?
    public let generatedAt: String?
    public let metrics: [String: Double]
}

/// Índice dos artefatos do repositório.
///
/// Lê `artifacts/**/*.json` direto — sem etapa de geração de código, sem
/// arquivo intermediário. O índice é a fonte, não uma cópia dela.
public struct Evidence: Sendable {
    public let artifacts: [String: Artifact]

    public init(artifactsRoot: URL) throws {
        var out: [String: Artifact] = [:]
        var skipped = 0

        let fm = FileManager.default
        guard let walker = fm.enumerator(at: artifactsRoot,
                                         includingPropertiesForKeys: [.isRegularFileKey]) else {
            throw EvidenceError.noArtifact(artifactsRoot.path)
        }

        for case let url as URL in walker where url.pathExtension == "json" {
            guard let data = try? Data(contentsOf: url),
                  let any = try? JSONSerialization.jsonObject(with: data),
                  let dict = any as? [String: Any] else { skipped += 1; continue }

            let id = url.path
                .replacingOccurrences(of: artifactsRoot.path + "/", with: "")
                .replacingOccurrences(of: ".json", with: "")

            // status às vezes vem como objeto: só string conta
            let status = (dict["status"] as? String) ?? (dict["gate"] as? String)
            let reason = dict["reason"] as? String
            let generatedAt = (dict["generated_at"] as? String) ?? (dict["timestamp"] as? String)

            var metrics: [String: Double] = [:]
            Self.flatten(dict["metrics"] as? [String: Any] ?? [:], into: &metrics)
            for (k, v) in dict where metrics[k] == nil {
                if let d = Self.number(v) { metrics[k] = d }
            }

            let (kind, level) = Self.classify(status)
            let prop = Self.wilson(metrics["passed"], metrics["total"])
            out[id] = Artifact(id: id, kind: kind, proportion: prop, territory: Self.territory(id),
                               status: status, level: level,
                               reason: reason, generatedAt: generatedAt, metrics: metrics)
        }
        self.artifacts = out
    }

    private static func flatten(_ obj: [String: Any], prefix: String = "",
                                into out: inout [String: Double]) {
        for (k, v) in obj {
            let key = prefix.isEmpty ? k : "\(prefix).\(k)"
            if let nested = v as? [String: Any] {
                flatten(nested, prefix: key, into: &out)
            } else if let d = number(v) {
                out[key] = d
            }
        }
    }

    /// Número real, não booleano.
    ///
    /// `v is Bool` NÃO serve: no Foundation do Linux ele devolve `true` para
    /// qualquer NSNumber valendo 0 ou 1, o que descartava em silêncio toda
    /// métrica igual a zero — justamente os "0 falhas" que este site existe
    /// para publicar. O objCType distingue: 'c' é booleano, o resto é número.
    private static func number(_ v: Any) -> Double? {
        guard let n = v as? NSNumber else { return nil }
        return String(cString: n.objCType) == "c" ? nil : n.doubleValue
    }

    /// Intervalo de Wilson 95%. "Todos passaram" com n=1 sustenta
    /// [0.207, 1.000] — quase nada. É o que a amostra realmente diz.
    private static func wilson(_ k: Double?, _ n: Double?, z: Double = 1.96) -> Proportion? {
        guard let k, let n, n > 0 else { return nil }
        let p = k / n
        let d = 1 + z * z / n
        let c = (p + z * z / (2 * n)) / d
        let h = z / d * (p * (1 - p) / n + z * z / (4 * n * n)).squareRoot()
        return Proportion(passed: Int(k), total: Int(n), point: p,
                          lo: max(0, c - h), hi: min(1, c + h))
    }

    /// Duas espécies, não uma com metade quebrada.
    private static func classify(_ status: String?) -> (Kind, Level?) {
        switch status?.lowercased() {
        case "pass": return (.gate, .verified)
        case "fail": return (.gate, .refused)
        case "partial", "beta", "active": return (.gate, .unbounded)
        default: return (.record, nil)
        }
    }

    /// Território é o primeiro segmento, com os sprintNN agrupados —
    /// sem isso são 88 territórios de um artefato cada.
    private static func territory(_ id: String) -> String {
        guard let i = id.firstIndex(of: "/") else { return "(raiz)" }
        let head = String(id[id.startIndex..<i])
        if head.hasPrefix("sprint"), head.dropFirst(6).allSatisfy(\.isNumber),
           !head.dropFirst(6).isEmpty {
            return "sprints"
        }
        return head
    }

    public func byLevel(_ l: Level) -> [Artifact] {
        artifacts.values.filter { $0.level == l }.sorted { $0.id < $1.id }
    }

    public func byKind(_ k: Kind) -> [Artifact] {
        artifacts.values.filter { $0.kind == k }.sorted { $0.id < $1.id }
    }

    public struct Territory: Sendable {
        public let name: String
        public let artifacts: [Artifact]
        public var gates: Int { artifacts.filter { $0.kind == .gate }.count }
        public var records: Int { artifacts.filter { $0.kind == .record }.count }
        public var refused: Int { artifacts.filter { $0.level == .refused }.count }
    }

    /// Territórios por tamanho; dentro de cada um, recusas primeiro.
    public var territories: [Territory] {
        Dictionary(grouping: artifacts.values, by: { $0.territory })
            .map { name, list in
                Territory(name: name, artifacts: list.sorted { a, b in
                    func rank(_ x: Artifact) -> Int {
                        x.level == .refused ? 0 : (x.kind == .gate ? 1 : 2)
                    }
                    return rank(a) != rank(b) ? rank(a) < rank(b) : a.id < b.id
                })
            }
            .sorted { $0.artifacts.count > $1.artifacts.count }
    }

    /// A recusa é resultado de primeira classe.
    public var refusals: [Artifact] { byLevel(.refused) }

    public struct Stats: Sendable {
        public let artifacts: Int, metrics: Int, gates: Int, records: Int, refused: Int
        public let proportions: Int, allPass: Int
        public let widest: Double, narrowest: Double
    }

    public var stats: Stats {
        var metrics = 0, gates = 0, records = 0, refused = 0
        var widths: [Double] = [], allPass = 0
        for a in artifacts.values {
            metrics += a.metrics.count
            if a.kind == .gate { gates += 1 } else { records += 1 }
            if a.level == .refused { refused += 1 }
            if let p = a.proportion {
                widths.append(p.width)
                if p.passed == p.total { allPass += 1 }
            }
        }
        widths.sort()
        return Stats(artifacts: artifacts.count, metrics: metrics, gates: gates,
                     records: records, refused: refused,
                     proportions: widths.count, allPass: allPass,
                     widest: widths.last ?? 0, narrowest: widths.first ?? 0)
    }

    /// 0 = a amostra restringe · 1 = a amostra mal restringe.
    public func tension(_ p: Proportion) -> Double {
        let s = stats
        let span = s.widest - s.narrowest
        return span > 0 ? (p.width - s.narrowest) / span : 0
    }
}
