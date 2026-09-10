# Deploy — next.souniolang.org

O gerador é escrito em Swift, e o contêiner de build da Vercel não tem
toolchain Swift. Por isso **o build acontece no GitHub Actions**, dentro da
imagem oficial `swift:6.0.3-noble`, e a Vercel recebe a saída pronta via
`vercel deploy --prebuilt` — que não executa build nenhum do lado dela.

O formato é a [Build Output API v3](https://vercel.com/docs/build-output-api/v3):
`scripts/build-vercel-output.mjs` monta `.vercel/output/` a partir de
`swift/dist/`.

## Build local

```bash
cd site-next/web  && npm ci && npm run build:islands
cd ../swift       && swift run sitegen
cd ..             && node scripts/build-vercel-output.mjs
```

Ou, do `site-next/web`, o ciclo curto: `npm run build:hybrid`.

## Configuração, uma vez só

O workflow existe e falha com mensagem clara enquanto estes três passos não
forem feitos. Nenhum deles foi executado por mim — criar projeto e apontar
domínio são ações na conta de vocês.

### 1. Criar o projeto na Vercel

Separado do `sounio` em produção (`prj_0FStNQ8BOUUiQHLP7D6AWah4mY8s`), que
continua servindo o site atual.

```bash
vercel project add sounio-next --scope team_rWORp5HA3dSzo95gt1RZd3sh
```

Sem framework e sem build command: tudo chega pronto. Deixe
`deploymentEnabled: false` para o git, como no projeto atual — o deploy é por
`workflow_dispatch` ou por tag `site-v2-*`.

### 2. Registrar o ID do projeto

Pegue o `prj_…` do projeto criado e defina como **variável de repositório**
(não secret — não é sigiloso, e assim aparece nos logs para conferência):

```
Settings → Secrets and variables → Actions → Variables
  VERCEL_PROJECT_ID_SITE_V2 = prj_…
```

O `VERCEL_TOKEN` já existe como secret desde o PR #2397.

### 3. Apontar o domínio

```bash
vercel domains add next.souniolang.org --scope team_rWORp5HA3dSzo95gt1RZd3sh
```

E o DNS: `next` como `CNAME` para `cname.vercel-dns.com`.

## Como se dispara

| Gatilho | Resultado |
|---|---|
| `workflow_dispatch` com `preview` | URL de preview |
| `workflow_dispatch` com `production` | vai para o domínio |
| tag `site-v2-*` | produção |

## As guardas

O workflow recusa continuar em três situações, todas com mensagem explícita:

- `VERCEL_TOKEN` ausente;
- `VERCEL_PROJECT_ID_SITE_V2` não definida;
- o projeto resolvido pelo `vercel pull` **não** ser o esperado — e há uma
  checagem específica que aborta se o ID resolvido for o do `sounio` em
  produção. Um deploy do site novo por cima do site atual seria o pior erro
  possível deste pipeline, e ele é impossível por construção.

O portão de evidência (`npm run gate`) roda antes do build. Um numeral sem
artefato no conteúdo derruba o deploy, não só o build local.

## O que ainda não está resolvido

- **A ilha carrega React?** Hoje não — `islands.js` tem 1,4 KB e é TypeScript
  puro. Quando entrarem simuladores three.js ou o seletor de idioma por página,
  React volta, e só nas páginas que os tiverem.
- **Cache dos artefatos.** O índice é regenerado a cada build a partir de
  `artifacts/**/*.json`. Com 523 arquivos leva menos de um segundo; se o corpus
  crescer muito, vale cachear por hash do diretório.
