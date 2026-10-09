import Foundation

public enum Level: String, Sendable {
    case verified, unbounded, refused
}

/// Um número que o site tem permissão para exibir.
///
/// O inicializador é `private`. Em Swift isso significa que nem este pacote,
/// fora deste arquivo, consegue fabricar um `Claim` — e nenhum consumidor
/// externo consegue de forma alguma. Não há cast, não há `as!`, não há
/// literal que produza um. A única origem é `Evidence.claim(_:metric:)`,
/// que exige um artefato existente e uma métrica que exista nele.
///
/// É aqui que o Swift ganha da versão em TypeScript: lá a marca de tipo
/// precisava de um `as Claim` dentro do módulo, e um cast é um furo. Aqui
/// o compilador fecha.
public struct Claim: Sendable {
    public let value: Double
    /// Null quando a origem é um registro.
    public let level: Level?
    public let artifact: String
    public let metric: String
    public let reason: String?
    public let generatedAt: String?

    private init(value: Double, level: Level?, artifact: String,
                 metric: String, reason: String?, generatedAt: String?) {
        self.value = value
        self.level = level
        self.artifact = artifact
        self.metric = metric
        self.reason = reason
        self.generatedAt = generatedAt
    }

    /// Fábrica interna ao arquivo — o único caminho, e ele é `fileprivate`.
    fileprivate static func make(from a: Artifact, metric: String, value: Double) -> Claim {
        Claim(value: value, level: a.level, artifact: a.id,
              metric: metric, reason: a.reason, generatedAt: a.generatedAt)
    }

    /// A língua-fonte do site é o inglês: 1,759 e não 1.759.
    /// Quando os locales entrarem, isto passa a receber o locale da página.
    ///
    /// `maximumFractionDigits` era 3, e isso publicava `1.99994` como `2` e
    /// `0.999999` como `1` — apagando em silêncio a diferença que a medição
    /// existe para mostrar. O portão garante a procedência do número; se o
    /// formatador o arredonda, a garantia não chega à página.
    public var formatted: String {
        let f = NumberFormatter()
        f.locale = Locale(identifier: "en_GB")
        f.numberStyle = .decimal
        f.maximumFractionDigits = value == value.rounded() ? 0 : 17
        return f.string(from: NSNumber(value: value)) ?? String(value)
    }

    public var glyph: String {
        switch level {
        case .verified: return "◈"
        case .unbounded: return "△"
        case .refused: return "⊘"
        case nil: return "·"   // registro: medição sem veredito
        }
    }
}

public enum EvidenceError: Error, CustomStringConvertible {
    case noArtifact(String)
    case noMetric(artifact: String, metric: String)

    public var description: String {
        switch self {
        case .noArtifact(let id):
            return "Sem evidência: não existe artefato \"\(id)\"."
        case .noMetric(let a, let m):
            return "Sem evidência: o artefato \"\(a)\" não expõe a métrica \"\(m)\"."
        }
    }
}

extension Evidence {
    /// A única porta de entrada para um número no site.
    /// Lança — e o gerador não trata: o build para.
    public func claim(_ artifactID: String, metric: String) throws -> Claim {
        guard let a = artifacts[artifactID] else {
            throw EvidenceError.noArtifact(artifactID)
        }
        guard let v = a.metrics[metric] else {
            throw EvidenceError.noMetric(artifact: artifactID, metric: metric)
        }
        return Claim.make(from: a, metric: metric, value: v)
    }
}
