---
title: "Epistemic Computing Manifesto"
description: "Computing at the boundary between what we know and what we don't"
layout: "manifesto"
subtitle: "Five Principles for Scientific Computing"
---

> *"The measure of intelligence is the ability to change."*
> — Albert Einstein

数十年にわたり、プログラミング言語は数値を完璧なものとして扱ってまいりました:`3.14159`とはまさにその値であり、それ以上でも以下でもありません。しかし、科学はこうした方法で機能するものではありません。すべての測定には誤差が伴います。すべてのモデルには不確実性が存在します。すべての予測には信頼区間が設定されます。

**Sounio**は、革新的な前提に基づいて構築されております:**不確実性はバグではなく—機能なのです**。

---

## Epistemic Computing(認識論的計算)の五原則

### 1. すべての知識は不確実である

物理世界において、完璧な測定など存在しません。ハイゼンベルクの不確定性原理は、私どもの計測機器の限界ではなく—現実の根本的な性質でございます。たとえ巨視的な測定であっても、ノイズ、キャリブレーション誤差、および有限の精度が伴います。

```sio
// Wrong: pretending we know exactly
let concentration = 5.23  // mg/L... but really?

// Right: acknowledging uncertainty
let concentration = Knowledge::new(5.23 mg/L, uncertainty: 0.15 mg/L)
```

Sounioでは、これを明確にいたします。値を宣言する際には、*この値をどの程度正確に把握しているのか?* を考慮しなければなりません。

### 2. Provenance(出所)は譲れないものである

出所のないデータは信頼のないデータでございます。規制機関が「この数値はどこから来たのですか?」と尋ねられた際には、一次資料まで遡る回答をお持ちであるべきです。

```sio
let clearance = Knowledge::new(
    value: 10.5 L/h,
    uncertainty: 1.2 L/h,
    source: Source {
        origin: "Phase III Trial NCT04123456",
        timestamp: 2025-03-15,
        method: "Population PK analysis",
        confidence: 0.95
    }
)
```

すべての`Knowledge<T>`は、そのprovenanceを携行いたします。データの系譜は、データそのものと同様に重要でございます。

### 3. 不確実性は自動的に伝播する

手動での不確実性伝播は煩雑で誤りやすいものです。GUM(Guide to the Expression of Uncertainty in Measurement:不確実性評価に関する国際標準ガイドライン)は、不確実性が数学的演算を通じてどのように結合するかを定義しております。Sounioでは、これを自動的に実装いたします。

```sio
let mass = Knowledge::new(100.0 g, uncertainty: 0.5 g)
let volume = Knowledge::new(50.0 mL, uncertainty: 0.2 mL)

// Density calculation with automatic propagation
let density = mass / volume
// density.uncertainty is computed via GUM:
// δρ/ρ = sqrt((δm/m)² + (δV/V)²)
```

物理学を記述なさるのはお客様でございます。コンパイラが統計処理を担います。

### 4. 信頼性が実行を制御する

すべての計算が盲目的に進むべきではありません。信頼性が閾値以下に低下した際には、実行を一時停止し、警告を発し、または代替経路を選択すべきです。

```sio
fn critical_decision(data: Knowledge<f64>) -> Action {
    if data.confidence < 0.90 {
        return Action::RequestMoreData
    }

    if data.confidence < 0.95 {
        return Action::ProceedWithCaution(data)
    }

    Action::Proceed(data)
}
```

これは防御的プログラミングではなく—*epistemic programming(認識論的プログラミング)*でございます。システムは自分が知らないことを認識しております。

### 5. 標準準拠を設計に組み込む

科学には標準が存在する理由がございます。Sounioは、以下の標準に準拠するよう構築されております:

- **GUM** — ISO Guide to the Expression of Uncertainty in Measurement(不確実性評価に関する国際標準ガイドライン)
- **ISO 17025** — Competence of testing and calibration laboratories(試験およびキャリブレーション研究所の能力に関する規格)
- **21 CFR Part 11** — Electronic records and signatures (FDA)(電子記録および電子署名に関するFDA規則)
- **FAIR Principles** — Findable, Accessible, Interoperable, Reusable data(発見可能、アクセス可能、相互運用可能、再利用可能データに関する原則)

これらは後付けではなく—建築的な基盤でございます。

---

## 私どもが解決する問題

### 再現性危機

2011年から2021年にかけて、米国だけで推定280億ドルが非再現性の前臨床研究に無駄に費やされました。原因は多岐にわたりますが、際立っておりますのは:**不確実性情報の喪失**でございます。

`5.23 mg/L`の測定値がシステム間で渡され、データベースに保存され、計算に使用される際—`±0.15`はしばしば消失いたします。下流の解析ではそれを正確な値として扱い、元の不確実性が排除していたであろう結論が導かれます。

### 解決策

Sounioは不確実性を*感染性*のものといたします。誤ってそれを落とすことはできません。型システムが、`Knowledge<T>`を単なる`T`に変換することを、明示的な承認なしには許しません。

```sio
let safe_value = measurement.value  // Compiler error!

let safe_value = measurement.unwrap_certain()  // Requires confidence > 0.99

let safe_value = measurement.acknowledge_uncertainty()  // Explicit opt-out, logged
```

---

## 「Sounio」とはなぜか?

アッティキの先端に位置するスニオン岬は、古代ギリシャの船乗りたちが地平線を見張った場所でございます。そこに立つポセイドン神殿は、目印であり祈りの場—不確実な海を航行するための固定点でございました。

Sounioという言語は、同じ目的を果たします:不確実なデータを航行するための安定した基盤でございます。柱はお客様の型システムです。海は科学領域でございます。地平線は確実性が終わり、探求が始まる場所です。

バイロンは1810年に訪れ、大理石に自身の名を刻みました(どうか真似なさらぬよう)。彼は次のように記しております:

> *"Place me on Sunium's marbled steep,*
> *Where nothing, save the waves and I,*
> *May hear our mutual murmurs sweep;*
> *There, swan-like, let me sing and die."*

私どもはそれほど劇的ではありません。しかし、あの柱のように長く残るものを構築しております。

---

## 今後の道筋

Sounioは完成しておりません。決して完成しないかもしれません。しかし、原則は定まっております:

1. **不確実性は第一級のもの** — ライブラリでも注釈でもなく、根本的な型でございます。

2. **伝播は正確** — GUM準拠、テスト済み、検証済み。

3. **Provenanceは保持される** — 出所から結果まで、鎖は途切れません。

4. **信頼性は行動可能** — システムは知っていることと知らないことに応答いたします。

5. **標準は組み込み** — 準拠はオプションではありません。

科学がより優れたツールに値するとお考えでしょうか—不確実性を無視するのではなく計算すべきだと—再現性が偶然ではなく機能であると—ならば、Sounioはお客様のためのものです。

---

*地平線でお会いいたしましょう。*

**🏛️ SOUNIO 🌊**
