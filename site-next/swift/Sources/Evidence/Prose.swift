import Foundation

/// Texto que o site tem permissão para publicar.
///
/// O inicializador recusa qualquer numeral que seja uma afirmação factual —
/// dois ou mais dígitos, percentagens, razões, decimais. Um número assim só
/// entra na página como `Claim`, isto é, com o artefato que o sustenta.
///
/// Em JS isto era um script que varria arquivos com regex depois do fato.
/// Aqui é o construtor: a prosa inválida não chega a existir, e o gerador
/// para no ponto exato onde alguém tentou afirmar sem evidência.
public struct Prose: Sendable {
    public let text: String

    private static let claimish = try! NSRegularExpression(
        pattern: #"(?<![\w.#-])(\d{2,}(?:[.,]\d+)?%?|\d+\s*/\s*\d+|\d+[.,]\d+)(?![\w%-])"#
    )

    /// Constantes do MÉTODO e identificadores, não resultados de medição.
    /// Espelha exatamente as exceções de scripts/gate-claims.mjs: se as duas
    /// implementações discordarem, a garantia do site depende de qual gerador
    /// foi usado — e aí não é garantia.
    private static let exempt: [NSRegularExpression] = [
        try! NSRegularExpression(pattern: #"\b95\s*%"#),
        try! NSRegularExpression(pattern: #"\bz\s*=\s*1\.96\b"#),
        try! NSRegularExpression(pattern: #"\b\d{1,2}:\d{2}\b"#),
        try! NSRegularExpression(pattern: #"\b\d{1,2}(?:[–-]\d{1,2})?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b"#,
                                 options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: #"\bE[0-5]\b"#),
        try! NSRegularExpression(pattern: #"\b(19|20)\d{2}\b"#),

        // Versões de nomes próprios de padrões e ferramentas. Explícita, não
        // genérica: uma regra do tipo "maiúscula seguida de número" isentaria
        // "Madaros 4.71" e seria o contrabando que este portão impede.
        try! NSRegularExpression(pattern: #"\bGRI-Mech\s+\d+(?:\.\d+)+"#),
        try! NSRegularExpression(pattern: #"\bCantera\s+\d+(?:\.\d+)+"#),
        try! NSRegularExpression(pattern: #"\bNASA-\d\b"#),
    ]

    public init(_ s: String) throws {
        // Mascara os trechos isentos e verifica o resto. Isentar a string
        // inteira porque ela menciona "95%" deixaria passar qualquer numeral
        // inventado no mesmo parágrafo.
        var masked = s
        for re in Self.exempt {
            let r = NSRange(masked.startIndex..<masked.endIndex, in: masked)
            masked = re.stringByReplacingMatches(in: masked, range: r, withTemplate: "·")
        }
        let mr = NSRange(masked.startIndex..<masked.endIndex, in: masked)
        if let m = Self.claimish.firstMatch(in: masked, range: mr),
           let rr = Range(m.range, in: masked) {
            throw ProseError.unbackedNumeral(String(masked[rr]), in: s)
        }
        self.text = s
    }

    /// Escape para textos onde o numeral não é afirmação — ids, caminhos,
    /// nomes de artefato. Explícito de propósito: quem usa assume.
    public static func verbatim(_ s: String) -> Prose {
        Prose(unchecked: s)
    }

    private init(unchecked s: String) { self.text = s }
}

public enum ProseError: Error, CustomStringConvertible {
    case unbackedNumeral(String, in: String)

    public var description: String {
        switch self {
        case .unbackedNumeral(let n, let s):
            return """
            Afirmação numérica sem evidência: "\(n)"
              em: \(s.prefix(96))
            Use claim(artefato, métrica) — ou Prose.verbatim se não for afirmação.
            """
        }
    }
}

// Deliberadamente SEM ExpressibleByStringLiteral.
//
// Com ele, `Prose("texto com 1759")` resolvia para o init de literal, que não
// lança — a violação virava fatalError e o `try` do chamador era decorativo.
// Sem ele, toda construção passa pelo init que lança, e o erro é tratável e
// diagnosticável. A conveniência do literal não vale a ambiguidade.
