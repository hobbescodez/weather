# Putting real access control on the dashboard

Written in response to: "I want real password protection on the GitHub Pages
site — not just a client-side JS prompt, since that's trivially bypassed."

That instinct is right, and it rules out more than just the JS prompt. Two
things have to be true before a login page means anything:

1. The gate can't be walked around.
2. The content isn't *also* readable somewhere the gate doesn't cover.

The current setup fails **both**, and the second one is the bigger problem.

---

## Blocker 1: the repo is public, so the dashboard is already readable

`hobbescodez/weather` is a public repo, and the rendered dashboard is committed
to it every hour at `docs/index.html`. So the full page is readable right now at:

    https://github.com/hobbescodez/weather/blob/claude/new-session-tlg3c4/docs/index.html

…along with `calibration_log.jsonl`, `daily_performance.jsonl`,
`paper_trades_pending.json`, and the rest of the logs.

**No access control in front of the Pages URL changes this.** Gating
`hobbescodez.github.io/weather/` while the repo stays public is bypassable in
exactly the same way the JS prompt is — you just read it from a different URL.
Fixing repo visibility is step zero; everything below assumes it's done.

## Blocker 2: Cloudflare Access cannot be applied to a `github.io` URL

Cloudflare Access applies to hostnames that are **an active zone in your own
Cloudflare account**. `github.io` is GitHub's domain — you can't add it to
Cloudflare, can't change its nameservers, and therefore can't put Access in
front of it. There is no free-tier trick around this; it's structural.

So "Cloudflare Access in front of the GitHub Pages URL" is not a thing that can
be built as literally described. The two real options are below.

---

## Option A — Cloudflare Pages + Access (recommended)

Move the static hosting to Cloudflare Pages. Access protects `*.pages.dev`
natively, so this needs **no domain purchase and no paid plan**. Because the
origin becomes Cloudflare rather than GitHub, the gate actually covers the
content instead of sitting beside it.

The hourly refresh Routine keeps working unchanged — it still commits and
pushes `docs/index.html`; Cloudflare Pages redeploys on each push.

### Steps you need to do (each needs your login — I can't do these for you)

**1. Make the GitHub repo private**
- <https://github.com/hobbescodez/weather/settings>
- Bottom of the page → *Danger Zone* → **Change repository visibility** →
  Private.
- This also turns off the existing GitHub Pages site, which is fine — you're
  replacing it. (Pages from a private repo would need GitHub Pro, ~$4/mo, and
  wouldn't fix Blocker 2 anyway. Skip it.)

**2. Create a Cloudflare account (free)**
- <https://dash.cloudflare.com/sign-up> — email + password, no card, no domain.

**3. Connect the repo to Cloudflare Pages**
- Cloudflare dashboard → **Workers & Pages** → *Create* → **Pages** tab →
  *Connect to Git*.
- Authorize Cloudflare's GitHub App, granting it access to `hobbescodez/weather`
  only (not all repos). Private repos are supported.
- Configure the build:
  - Production branch: `claude/new-session-tlg3c4`
  - Framework preset: **None**
  - Build command: *(leave empty)*
  - Build output directory: `docs`
- Save and deploy. You'll get a URL like `weather-abc.pages.dev`.
- Confirm the dashboard renders there before continuing.

**4. Turn on Access**
- Cloudflare dashboard → **Zero Trust**. First visit asks you to pick a team
  name (becomes `yourteam.cloudflareaccess.com`) and a plan — choose the
  **Free** plan. It may ask for a card to verify; free tier covers 50 users and
  won't be charged.
- **Access controls → Applications → Add an application → Self-hosted.**
- Application name: `Weather dashboard`
- Add the public hostname: your `*.pages.dev` domain from step 3.
- **Add a policy:**
  - Name: `Just me`
  - Action: **Allow**
  - Include → **Emails** → your email address.
- Save.

**5. Verify the gate actually holds**
- Open the `pages.dev` URL in a private window → you should get a Cloudflare
  login screen, not the dashboard.
- Enter your email → Cloudflare sends a one-time PIN → enter it → dashboard
  loads.
- With JS disabled, and via `curl`, the URL should still return the login
  challenge rather than the page. That's the test the JS prompt failed.

### On authentication method

The default is a one-time PIN emailed to you — no password to manage, and it's
genuinely server-side. If you'd rather click "Sign in with Google" than wait for
an email, add a login method under **Zero Trust → Settings → Authentication →
Login methods**. Not required.

---

## Option B — keep GitHub Pages, add a custom domain

Only worth it if you specifically want the site on your own domain.

- Buy a domain (~$10–15/yr — this is the part that isn't free).
- Add it to Cloudflare, move its nameservers there.
- Point it at GitHub Pages, set it as the custom domain in the repo's Pages
  settings, proxy the DNS record (orange cloud), then add an Access application
  for that hostname.

**Caveats, because they matter here:**
- GitHub needs the record temporarily un-proxied (grey cloud) to issue its
  certificate; you proxy it afterward. Fiddly but documented.
- The repo still has to be private, or Blocker 1 stands.
- With a private repo you'd need GitHub Pro for Pages to publish at all.

Option A avoids all three. I'd only take Option B for the vanity domain.

---

## What I did not do

Nothing above was executed. Every step needs your GitHub or Cloudflare
credentials, and I'm not going to click through account creation or flip repo
visibility on your behalf. The one thing I *did* change is unrelated to access
control: the git remote and the NWS `User-Agent` string now say `hobbescodez`
instead of the old username.

## Sources

- [Cloudflare — publish a self-hosted application](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/self-hosted-public-app/)
  ("Domains must belong to an active zone in your Cloudflare account.")
- [Cloudflare Pages — custom domains & Access policies](https://developers.cloudflare.com/pages/configuration/custom-domains/)
- [GitHub — Pages availability by plan](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
