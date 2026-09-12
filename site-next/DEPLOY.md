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
| 1. Criar o projeto | ⚠️ criado e **PAUSADO** — `prj_8hvPYrL97qWYsCfeZp1wkxR5Gp5E` |
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

### O projeto está PAUSADO, e por quê

Ele nasceu com **deploy por git ligado**, e a consequência foi maior do que eu
previ. Em segundos a Vercel varreu o repositório e criou **quatro deploys**:
dois da `main` e de `fix/gum-2cov` marcados `target: production`, e dois
previews. O bot da Vercel comentou no **PR #2477, de outra pessoa**, que não
tem nenhuma relação com este trabalho.

Eu previ "builds que falham, ruído"; na verdade eles **tiveram sucesso** e
espalharam comentários em PRs alheios. Pausei o projeto para estancar
(`pause_project`), o que impede novos deploys por git.

Nada disso alcançou produção: o `sounio` foi comparado antes e depois —
`updatedAt`, último deploy, domínios e framework idênticos.

**Antes de despausar**, desligue o deploy por git no painel:
**Project → Settings → Git** — desconecte o repositório, ou use *Ignored Build
Step* com `exit 0`. Só então `unpause`, porque o deploy real é `--prebuilt` por
`workflow_dispatch` ou tag e não precisa da ligação com o git.

Não resolva com um `vercel.json` na raiz: o `sounio` de produção está ligado ao
**mesmo** repositório, e uma configuração de raiz pode alcançá-lo. Além disso
não adiantaria — um arquivo no meu branch não muda o comportamento da `main`
nem dos PRs alheios.

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
