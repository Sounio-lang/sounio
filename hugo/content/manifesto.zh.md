---
title: "Epistemic Computing Manifesto"
description: "Computing at the boundary between what we know and what we don't"
layout: "manifesto"
subtitle: "Five Principles for Scientific Computing"
---

> *"The measure of intelligence is the ability to change."*
> — Albert Einstein

几十年来,编程语言一直将数字视为完美的:`3.14159` 就是这样,没有多,也没有少。但科学并非如此运作。每一次测量都带有误差。每一种模型都存在不确定性。每一次预测都有置信界限。

**Sounio** 建立在一个激进的前提之上:**不确定性不是 bug——它是特性**。

---

## Epistemic Computing(认识计算)的五大原则

### 1. 所有知识都是不确定的

在物理世界中,没有完美的测量存在。海森堡不确定性原理并非我们仪器的局限——它是现实的基本属性。即使是宏观测量也带有噪声、校准误差和有限精度。

```sio
// 错误:假装我们确切知道
let concentration = 5.23  // mg/L... 但真的是吗?

// 正确:承认不确定性
let concentration = Knowledge::new(5.23 mg/L, uncertainty: 0.15 mg/L)
```

Sounio 使这一点显而易见。当你声明一个值时,你必须考虑:*我实际知道这个值有多准确?*

### 2. Provenance(溯源)是不可谈判的

没有来源的数据就是没有信任的数据。当监管机构询问"这个数字从何而来?"时,你应该有一个答案,能够追溯到主要来源。

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

每一个 `Knowledge<T>` 都携带其 provenance(溯源)。你数据的血统与数据本身同样重要。

### 3. 不确定性自动传播

手动不确定性传播既繁琐又容易出错。GUM(不确定性表达指南,Guide to the Expression of Uncertainty in Measurement)定义了不确定性如何通过数学运算组合。Sounio 自动实现这一点。

```sio
let mass = Knowledge::new(100.0 g, uncertainty: 0.5 g)
let volume = Knowledge::new(50.0 mL, uncertainty: 0.2 mL)

// 带有自动传播的密度计算
let density = mass / volume
// density.uncertainty 通过 GUM 计算:
// δρ/ρ = sqrt((δm/m)² + (δV/V)²)
```

你编写物理学。编译器处理统计学。

### 4. 置信度控制执行

并非所有计算都应盲目进行。当置信度低于阈值时,执行应暂停、警告或采取替代路径。

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

这不是防御性编程——这是 *epistemic programming(认识编程)*。系统知道自己不知道什么。

### 5. 标准合规性由设计决定

科学有标准是有原因的。Sounio 被构建为符合以下标准:

- **GUM** — ISO Guide to the Expression of Uncertainty in Measurement(不确定性表达的 ISO 指南)
- **ISO 17025** — Competence of testing and calibration laboratories(测试和校准实验室的能力)
- **21 CFR Part 11** — Electronic records and signatures (FDA)(电子记录和签名,FDA)
- **FAIR Principles** — Findable, Accessible, Interoperable, Reusable data(可发现、可访问、可互操作、可重用数据)

这些不是事后添加——它们是架构基础。

---

## 我们正在解决的问题

### 可重复性危机

在 2011 年至 2021 年间,仅在美国,就有估计 280 亿美元浪费在不可重复的临床前研究上。原因众多,但一个突出:**不确定性信息的丢失**。

当一个 `5.23 mg/L` 的测量值在系统间传递、在数据库中存储并用于计算时——`±0.15` 往往消失。下游分析将其视为确切值。得出的结论是原不确定性本会排除的。

### 解决方案

Sounio 使不确定性 *传染性*。你无法意外丢弃它。类型系统不会允许你将 `Knowledge<T>` 转换为裸露的 `T`,除非明确承认。

```sio
let safe_value = measurement.value  // 编译器错误!

let safe_value = measurement.unwrap_certain()  // 需要置信度 > 0.99

let safe_value = measurement.acknowledge_uncertainty()  // 明确选择退出,并记录日志
```

---

## 为什么叫"Sounio"?

苏尼翁角位于阿提卡的尖端,是古代希腊水手瞭望地平线的地方。那里的波塞冬神庙既是地标,也是一座祈祷——一个从不确定之海中导航的固定点。

Sounio 这一语言服务于相同目的:为导航不确定数据提供稳定基础。柱子是你的类型系统。大海是你的科学领域。地平线是确定性结束、探索开始的地方。

拜伦勋爵于 1810 年访问那里,并在大理石上刻下他的名字(请不要这样做)。他写道:

> *"Place me on Sunium's marbled steep,*
> *Where nothing, save the waves and I,*
> *May hear our mutual murmurs sweep;*
> *There, swan-like, let me sing and die."*

我们没有那么戏剧化。但我们正在构建一些东西,就像那些柱子一样,或许能经久不衰。

---

## 前路

Sounio 尚未完成。它可能永远不会完成。但原则已定:

1. **不确定性是一等公民** — 不是库,不是注解,而是基本类型。

2. **传播是正确的** — 符合 GUM,经测试、验证。

3. **Provenance(溯源)被保留** — 从来源到结果,链条不破。

4. **置信度是可行动的** — 系统响应它知道和不知道的东西。

5. **标准内置** — 合规性不是可选的。

如果你相信科学值得更好的工具——不确定性应被计算,而非忽略——可重复性是特性,而非意外——那么 Sounio 就是为你而建的。

---

*加入我们,前往地平线。*

**🏛️ SOUNIO 🌊**
