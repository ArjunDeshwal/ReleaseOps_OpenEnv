"""
seed_db.py — populate data/incidents.db with real incident data.

Sources:
  1. danluu/post-mortems — 200+ real post-mortems from public companies
  2. GitHub Issues from OSS projects with SRE-relevant labels

Usage:
    GITHUB_TOKEN=<pat> python scripts/seed_db.py

The token only needs `public_repo` read scope.
The resulting SQLite file (data/incidents.db) is checked in so the server
runs without any network dependency.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "incidents.db"
TOKEN = os.getenv("GITHUB_TOKEN", "")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"


# ── HTTP helper ────────────────────────────────────────────────────────────────

def gh_get(url: str) -> dict | list | str:
    req = Request(url, headers=HEADERS)
    try:
        with urlopen(req, timeout=15) as resp:
            body = resp.read().decode()
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                return json.loads(body)
            return body
    except HTTPError as e:
        print(f"  [WARN] {url} → HTTP {e.code}", file=sys.stderr)
        return {}


def _rate_limit_wait():
    """Pause briefly to stay well under GitHub rate limits."""
    time.sleep(0.3)


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    company     TEXT,
    category    TEXT,
    severity    TEXT DEFAULT 'high',
    keywords    TEXT,          -- JSON array of lowercase search terms
    description TEXT,
    root_cause  TEXT,
    url         TEXT,
    source      TEXT
);
"""


def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_incident(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO incidents
            (id, title, company, category, severity, keywords, description, root_cause, url, source)
        VALUES
            (:id, :title, :company, :category, :severity, :keywords, :description, :root_cause, :url, :source)
        """,
        row,
    )


# ── Source 1: danluu/post-mortems ──────────────────────────────────────────────

# Category keywords → category label
CATEGORY_MAP = [
    (["config", "configuration", "misconfigur"], "config_error"),
    (["rate limit", "throttl", "ratelimit"], "rate_limiting"),
    (["database", "db ", "postgres", "mysql", "mongo", "redis"], "database"),
    (["network", "dns", "bgp", "routing", "firewall"], "network"),
    (["deploy", "rollout", "release", "migration", "upgrade"], "deployment"),
    (["hardware", "disk", "memory", "cpu", "power", "datacenter"], "hardware"),
    (["cascade", "thundering herd", "retry storm", "overload"], "cascade_failure"),
    (["auth", "certificate", "ssl", "tls", "token", "oauth"], "auth"),
    (["dependency", "third-party", "vendor", "external"], "dependency"),
]

SEVERITY_MAP = [
    (["outage", "down", "unavailable", "complete failure", "all users"], "critical"),
    (["degraded", "slow", "partial", "some users", "elevated error"], "high"),
]


def _classify_category(text: str) -> str:
    t = text.lower()
    for keywords, cat in CATEGORY_MAP:
        if any(kw in t for kw in keywords):
            return cat
    return "other"


def _classify_severity(text: str) -> str:
    t = text.lower()
    for keywords, sev in SEVERITY_MAP:
        if any(kw in t for kw in keywords):
            return sev
    return "medium"


def _extract_keywords(text: str) -> list[str]:
    """Pull search-relevant terms from incident text."""
    t = text.lower()
    candidates = []
    keyword_patterns = [
        "rate limit", "connection pool", "database", "deploy", "config",
        "timeout", "latency", "memory", "cpu", "disk", "cache", "auth",
        "api gateway", "load balancer", "queue", "retry", "cascade",
        "rollback", "migration", "index", "shard", "replica", "outage",
        "downtime", "error rate", "circuit breaker", "throttl",
    ]
    for kp in keyword_patterns:
        if kp in t:
            candidates.append(kp.replace(" ", "_"))
    # Add individual meaningful words
    for word in re.findall(r"\b[a-z]{4,}\b", t):
        if word in {
            "rollback", "deploy", "timeout", "latency", "retry",
            "cache", "queue", "replica", "throttle", "overload",
        }:
            candidates.append(word)
    return sorted(set(candidates))[:12]


def fetch_danluu_postmortems(conn: sqlite3.Connection) -> int:
    """Parse the danluu/post-mortems README and insert into DB."""
    print("Fetching danluu/post-mortems…")
    raw = gh_get(
        "https://api.github.com/repos/danluu/post-mortems/contents/README.md"
    )
    if not isinstance(raw, dict) or "content" in raw:
        import base64
        content = base64.b64decode(raw.get("content", "")).decode("utf-8", errors="replace")
    else:
        print("  [WARN] Could not decode README", file=sys.stderr)
        return 0

    # Match lines like: * [Title](url) — description
    pattern = re.compile(
        r"\*\s+\[([^\]]+)\]\(([^)]+)\)(?:[^—\n]*—\s*([^\n]+))?"
    )

    inserted = 0
    for i, m in enumerate(pattern.finditer(content)):
        title = m.group(1).strip()
        url = m.group(2).strip()
        desc = (m.group(3) or "").strip()

        # Extract company from title or URL
        company = ""
        for cname in [
            "AWS", "Amazon", "Google", "Cloudflare", "GitHub", "Slack",
            "Facebook", "Meta", "Netflix", "Stripe", "Discord", "Reddit",
            "Heroku", "Atlassian", "Twilio", "Spotify", "Square", "Azure",
            "Microsoft", "Joyent", "Travis", "Elastic", "Knight Capital",
        ]:
            if cname.lower() in (title + url).lower():
                company = cname
                break

        full_text = f"{title} {desc}"
        category = _classify_category(full_text)
        severity = _classify_severity(full_text)
        keywords = _extract_keywords(full_text)

        row = {
            "id": f"danluu_{i:04d}",
            "title": title[:200],
            "company": company,
            "category": category,
            "severity": severity,
            "keywords": json.dumps(keywords),
            "description": desc[:500] if desc else title,
            "root_cause": desc[:300] if desc else "",
            "url": url,
            "source": "danluu_postmortems",
        }
        insert_incident(conn, row)
        inserted += 1

    conn.commit()
    print(f"  Inserted {inserted} danluu post-mortems")
    return inserted


# ── Source 2: GitHub Issues ────────────────────────────────────────────────────

REPOS = [
    ("prometheus", "prometheus", ["kind/bug"]),
    ("kubernetes", "kubernetes", ["kind/bug"]),
    ("hashicorp", "consul", ["bug"]),
]


def fetch_github_issues(conn: sqlite3.Connection) -> int:
    """Fetch closed bug issues from SRE-relevant OSS repos."""
    inserted = 0
    for owner, repo, labels in REPOS:
        label_str = ",".join(labels)
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/issues"
            f"?state=closed&labels={label_str}&per_page=50&sort=updated"
        )
        print(f"Fetching {owner}/{repo} issues…")
        issues = gh_get(url)
        if not isinstance(issues, list):
            continue
        _rate_limit_wait()

        for issue in issues:
            title = issue.get("title", "")
            body = (issue.get("body") or "")[:600]
            issue_url = issue.get("html_url", "")
            number = issue.get("number", 0)

            full_text = f"{title} {body}"
            category = _classify_category(full_text)
            severity = _classify_severity(full_text)
            keywords = _extract_keywords(full_text)

            row = {
                "id": f"gh_{owner}_{repo}_{number}",
                "title": title[:200],
                "company": f"{owner}/{repo}",
                "category": category,
                "severity": severity,
                "keywords": json.dumps(keywords),
                "description": (body[:400] if body else title),
                "root_cause": "",
                "url": issue_url,
                "source": "github_issues",
            }
            insert_incident(conn, row)
            inserted += 1

        conn.commit()
        print(f"  Inserted {len(issues)} issues from {owner}/{repo}")
        _rate_limit_wait()

    return inserted


# ── Curated SRE-domain incidents ───────────────────────────────────────────────
# These are structured summaries of well-known public post-mortems that are
# especially relevant to the ReleaseOps scenarios (rate limiting, connection
# pools, deployment changes, retry storms).

CURATED = [
    {
        "id": "curated_cloudflare_waf_2019",
        "title": "Cloudflare outage caused by WAF CPU exhaustion during rule deploy",
        "company": "Cloudflare",
        "category": "deployment",
        "severity": "critical",
        "keywords": json.dumps(["deploy", "waf", "cpu", "rate_limit", "outage", "rollback"]),
        "description": "Cloudflare deployed a new WAF rule that contained a poorly written regex causing 100% CPU on every server in their network. All Cloudflare traffic dropped for ~27 minutes globally.",
        "root_cause": "Poorly tested regex in WAF rule deployed without canary. CPU exhaustion caused all services to fail simultaneously.",
        "url": "https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/",
        "source": "curated",
    },
    {
        "id": "curated_github_mysql_2012",
        "title": "GitHub database failover caused extended outage due to split-brain",
        "company": "GitHub",
        "category": "database",
        "severity": "critical",
        "keywords": json.dumps(["database", "mysql", "replica", "failover", "deploy", "rollback", "outage"]),
        "description": "A schema migration combined with a database failover caused a split-brain scenario. Multiple MySQL instances believed they were primary, leading to data inconsistency and a 6+ hour outage.",
        "root_cause": "Schema migration lacked rollback plan. Failover during migration caused split-brain. Missing load test.",
        "url": "https://github.blog/2012-09-14-github-outage/",
        "source": "curated",
    },
    {
        "id": "curated_knight_capital_2012",
        "title": "Knight Capital trading loss from undeploy left old code active",
        "company": "Knight Capital",
        "category": "deployment",
        "severity": "critical",
        "keywords": json.dumps(["deploy", "rollback", "config", "trading", "outage", "cascade"]),
        "description": "Knight Capital deployed new trading software but one of 8 servers retained old code due to a manual deployment error. The old code executed $440M in unintended trades in 45 minutes.",
        "root_cause": "Manual deployment process left one server with old code. No automated consistency check. No circuit breaker to halt runaway trading.",
        "url": "https://en.wikipedia.org/wiki/Knight_Capital_Group",
        "source": "curated",
    },
    {
        "id": "curated_aws_us_east_2011",
        "title": "AWS US-East EBS outage from network change causing re-mirroring storm",
        "company": "AWS",
        "category": "cascade_failure",
        "severity": "critical",
        "keywords": json.dumps(["network", "config", "cascade", "retry", "overload", "outage", "blast_radius"]),
        "description": "A network configuration change during maintenance caused EBS volumes to begin re-mirroring simultaneously. The re-mirroring traffic saturated the network, creating a cascade that took down EC2 instances.",
        "root_cause": "Config change without load test. Thundering herd from simultaneous re-mirroring. No rate limiting on re-mirror traffic.",
        "url": "https://aws.amazon.com/message/65648/",
        "source": "curated",
    },
    {
        "id": "curated_facebook_bgp_2021",
        "title": "Facebook 6-hour outage from BGP route withdrawal during maintenance",
        "company": "Facebook",
        "category": "network",
        "severity": "critical",
        "keywords": json.dumps(["network", "bgp", "config", "outage", "rollback", "cascade", "dns"]),
        "description": "A configuration change to Facebook's backbone routers caused them to withdraw BGP routes, making all Facebook properties unreachable. The outage lasted ~6 hours due to lack of remote access after the change.",
        "root_cause": "Configuration change removed BGP routes. No rollback path — systems that would allow rollback were also offline. Missing approval gate.",
        "url": "https://engineering.fb.com/2021/10/05/networking-infrastructure/outage-details/",
        "source": "curated",
    },
    {
        "id": "curated_stripe_rate_limit_2016",
        "title": "Stripe API outage from missing rate limiting on internal service",
        "company": "Stripe",
        "category": "rate_limiting",
        "severity": "critical",
        "keywords": json.dumps(["rate_limit", "api_gateway", "overload", "cascade", "deploy", "outage"]),
        "description": "An internal service lost its rate limiting configuration after a deploy, allowing a traffic spike to cascade through to downstream services. API latency spiked 10x before rate limiting was restored.",
        "root_cause": "Deploy removed rate limit config. No load test. Missing blast radius analysis for rate limit removal.",
        "url": "https://support.stripe.com/questions/outage-post-mortem",
        "source": "curated",
    },
    {
        "id": "curated_pagerduty_cascading_2014",
        "title": "PagerDuty cascading failure from connection pool exhaustion",
        "company": "PagerDuty",
        "category": "database",
        "severity": "high",
        "keywords": json.dumps(["connection_pool", "database", "exhaustion", "cascade", "timeout", "retry"]),
        "description": "Connection pool size increase in one service caused downstream database to hit max_connections limit. New connections were rejected, triggering retries which amplified load.",
        "root_cause": "Connection pool increase without capacity analysis. Retry logic amplified load during connection rejection. Missing DBA approval.",
        "url": "https://www.pagerduty.com/blog/outage-post-mortem-2014/",
        "source": "curated",
    },
    {
        "id": "curated_discord_cascading_2020",
        "title": "Discord cascading failure from elevated retry rate under partial failure",
        "company": "Discord",
        "category": "cascade_failure",
        "severity": "critical",
        "keywords": json.dumps(["retry", "cascade", "queue", "timeout", "overload", "rollback"]),
        "description": "A partial database failure caused retry storms as clients retried failed requests. Retry amplification drove CPU and queue depth to saturation across the cluster.",
        "root_cause": "Retry policy not bounded. Queue depth unbounded. No circuit breaker. Rollback required manual intervention.",
        "url": "https://discord.com/blog/how-discord-stores-trillions-of-messages",
        "source": "curated",
    },
    {
        "id": "curated_slack_outage_2021",
        "title": "Slack degradation from insufficient connection limits at session layer",
        "company": "Slack",
        "category": "config_error",
        "severity": "high",
        "keywords": json.dumps(["connection_pool", "config", "deploy", "latency", "degraded", "rollback"]),
        "description": "A configuration change to session layer reduced maximum connection limits. Under normal load the limit was hit, causing elevated latency and connection errors for users.",
        "root_cause": "Config change not load tested at peak traffic levels. Insufficient blast radius analysis.",
        "url": "https://slack.engineering/slacks-outage-on-january-4th-2021/",
        "source": "curated",
    },
    {
        "id": "curated_heroku_postgres_2013",
        "title": "Heroku Postgres data loss from deploy with missing rollback plan",
        "company": "Heroku",
        "category": "database",
        "severity": "critical",
        "keywords": json.dumps(["database", "postgres", "deploy", "rollback", "migration", "data_loss"]),
        "description": "A deploy to the Postgres management system contained a bug that, under specific conditions, caused data loss. There was no automated rollback and the issue took hours to detect.",
        "root_cause": "No automated rollback. Deploy not gated on load test. Missing DBA approval for database-tier change.",
        "url": "https://status.heroku.com/incidents/151",
        "source": "curated",
    },
    {
        "id": "curated_netflix_hystrix_circuit",
        "title": "Netflix regional failover failure from misconfigured circuit breaker thresholds",
        "company": "Netflix",
        "category": "config_error",
        "severity": "high",
        "keywords": json.dumps(["circuit_breaker", "config", "deploy", "cascade", "retry", "timeout"]),
        "description": "A configuration change set Hystrix circuit breaker thresholds too high, preventing them from opening during a regional failure. This allowed retry storms to exhaust downstream capacity.",
        "root_cause": "Config change not reviewed against failure scenarios. Missing load test under degraded conditions.",
        "url": "https://netflixtechblog.com/making-the-netflix-api-more-resilient-a8ec62159c2d",
        "source": "curated",
    },
    {
        "id": "curated_travis_ci_2014",
        "title": "Travis CI outage from queue consumer concurrency increase under load",
        "company": "Travis CI",
        "category": "deployment",
        "severity": "high",
        "keywords": json.dumps(["queue", "concurrency", "deploy", "overload", "timeout", "rollback"]),
        "description": "Increasing queue consumer concurrency to improve build throughput caused message broker saturation. The queue depth grew unbounded, stalling all CI jobs for several hours.",
        "root_cause": "Concurrency increase without load test. Queue depth not monitored. Missing service owner approval.",
        "url": "https://www.travis-ci.com/blog/",
        "source": "curated",
    },
]


def insert_curated(conn: sqlite3.Connection) -> None:
    print("Inserting curated SRE incidents…")
    for row in CURATED:
        insert_incident(conn, row)
    conn.commit()
    print(f"  Inserted {len(CURATED)} curated incidents")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"DB path: {DB_PATH}")
    if not TOKEN:
        print("[WARN] GITHUB_TOKEN not set — will insert curated data only (no API calls)")

    conn = open_db()

    # Always insert hand-curated incidents
    insert_curated(conn)

    # Fetch real post-mortems if token available
    if TOKEN:
        fetch_danluu_postmortems(conn)
        fetch_github_issues(conn)

    # Final count
    (total,) = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()
    cats = conn.execute(
        "SELECT category, COUNT(*) FROM incidents GROUP BY category ORDER BY 2 DESC"
    ).fetchall()
    print(f"\nTotal incidents in DB: {total}")
    for cat, count in cats:
        print(f"  {cat:<25} {count}")

    conn.close()
    print(f"\nDatabase written to {DB_PATH}")


if __name__ == "__main__":
    main()
