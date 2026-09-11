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

## Configuração

**Estado: dois dos três passos feitos.**

| Passo | Estado |
|---|---|
| 1. Criar o projeto | ✅ `sounio-next` = `prj_8hvPYrL97qWYsCfeZp1wkxR5Gp5E` |
| 2. Registrar o ID | ✅ variável `VERCEL_PROJECT_ID_SITE_V2` definida |
| 3. Apontar o domínio | ⬜ pendente — ação no DNS de vocês |

Restam duas coisas, ambas fora do que dá para fazer por API sem um token de
escopo amplo:

- **desligar o deploy por git** no `sounio-next` (ver abaixo);
- apontar `next.souniolang.org`.

O projeto foi criado **sem framework e sem deploy** (`latestDeployment: null`),
como especificado. Antes e depois da criação, o `sounio` em produção foi
fotografado e comparado: `updatedAt`, último deploy, domínios e framework
idênticos — **não foi tocado**.

### 1. Criar o projeto na Vercel — feito

`sounio-next` = `prj_8hvPYrL97qWYsCfeZp1wkxR5Gp5E`, separado do `sounio` em
produção (`prj_0FStNQ8BOUUiQHLP7D6AWah4mY8s`), que continua servindo o site
atual. Criado sem framework e sem deploy.

**Pendente e importante:** ele ficou com **deploy por git ligado**. Como está
ligado a `Sounio-lang/sounio`, todo push na `main` vai disparar um build na
Vercel — que **vai falhar**, porque o contêiner dela não tem Swift e o projeto
não tem framework. Não é perigoso (o deploy real é `--prebuilt`, por
`workflow_dispatch` ou tag, e não passa por esse caminho), mas é ruído a cada
push.

Desligar no painel: **Project → Settings → Git → Ignored Build Step**, ou
desconectar o repositório. Evite resolver com um `vercel.json` na raiz: o
`sounio` de produção está ligado ao **mesmo** repositório, e uma configuração
de raiz pode alcançá-lo.

### 2. Registrar o ID do projeto — feito

```
VERCEL_PROJECT_ID_SITE_V2 = prj_8hvPYrL97qWYsCfeZp1wkxR5Gp5E
```

Variável de repositório, não secret — não é sigiloso, e assim aparece nos logs
para conferência. O `VERCEL_TOKEN` já existia como secret desde o PR #2397.

A guarda do workflow aprova este ID: ela aborta só se o projeto resolvido for
`prj_0FStNQ8BOUUiQHLP7D6AWah4mY8s`.

### 3. Apontar o domínio — pendente

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
