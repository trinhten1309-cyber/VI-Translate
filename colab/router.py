"""Failover router: try providers in priority order, switch when one runs out.

- 429 that mentions a daily quota  -> provider disabled for the rest of the session
- 429 per-minute                   -> wait (Retry-After, capped) and retry; after
                                      MAX_429 in a row the provider cools down COOLDOWN s
- 401/403 (bad key), 404 (model renamed) -> provider disabled, reason printed
- timeout / connection / 5xx       -> short pause, then next provider
"""
import re
import time

import openai

DAILY = re.compile(r"per.?day|daily|PerDay|\bRPD\b|\bTPD\b|free-models-per-day", re.I)


class Router:
    MAX_429 = 2        # per-minute 429s in a row before cooling a provider down
    COOLDOWN = 600     # seconds a cooled-down provider is skipped
    MAX_WAIT = 65      # longest single wait for a per-minute limit

    def __init__(self, providers, sleep=time.sleep, clock=time.time, log=print):
        # providers: list of dicts {name, client, model}; list order = priority
        self.providers = providers
        self.state = {p["name"]: {"off": None, "until": 0.0, "hits": 0, "used": 0} for p in providers}
        self.sleep, self.clock, self.log = sleep, clock, log

    def available(self):
        now = self.clock()
        return [p for p in self.providers
                if not self.state[p["name"]]["off"] and self.state[p["name"]]["until"] <= now]

    def _disable(self, name, why):
        self.state[name]["off"] = why
        self.log(f"⛔ {name}: tắt cho phiên này — {why}")

    def chat(self, messages, max_attempts=12, **kw):
        """Return (text, provider_name). Raises RuntimeError when every provider is out
        or the request keeps failing (max_attempts calls)."""
        attempts = 0
        while True:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Yêu cầu thất bại sau {max_attempts} lần thử")
            live = self.available()
            if not live:
                cooling = [s["until"] for s in self.state.values() if not s["off"]]
                if not cooling:
                    raise RuntimeError("Tất cả nguồn đã hết lượt hoặc lỗi: " +
                                       "; ".join(f"{n}={s['off']}" for n, s in self.state.items()))
                wait = max(1.0, min(cooling) - self.clock())
                self.log(f"⏳ Mọi nguồn đang tạm nghỉ, chờ {wait:.0f}s")
                self.sleep(wait)
                continue
            p = live[0]
            st = self.state[p["name"]]
            try:
                out = p["client"].chat.completions.create(model=p["model"], messages=messages, **kw)
                st["hits"] = 0
                st["used"] += 1
                return (out.choices[0].message.content or "").strip(), p["name"]
            except openai.RateLimitError as e:
                msg = str(e)
                if DAILY.search(msg):
                    self._disable(p["name"], "hết hạn mức trong ngày")
                    continue
                st["hits"] += 1
                if st["hits"] >= self.MAX_429:
                    st["until"], st["hits"] = self.clock() + self.COOLDOWN, 0
                    self.log(f"💤 {p['name']}: bị giới hạn liên tục, nghỉ {self.COOLDOWN // 60} phút")
                    continue
                retry = e.response.headers.get("retry-after") if getattr(e, "response", None) else None
                wait = min(float(retry), self.MAX_WAIT) if retry and retry.replace(".", "").isdigit() else 20
                self.log(f"⏳ {p['name']}: giới hạn theo phút, chờ {wait:.0f}s")
                self.sleep(wait)
            except (openai.AuthenticationError, openai.PermissionDeniedError):
                self._disable(p["name"], "key sai hoặc không có quyền (401/403)")
            except openai.NotFoundError:
                self._disable(p["name"], f"không tìm thấy mô hình '{p['model']}' (404)")
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
                st["until"] = self.clock() + 60
                self.log(f"⚠️ {p['name']}: {type(e).__name__}, thử nguồn khác")
            except openai.APIStatusError as e:
                # e.g. 400 bad request, 413 request too large for this provider's limits
                st["until"] = self.clock() + 300
                self.log(f"⚠️ {p['name']}: lỗi {e.status_code} — {str(e)[:120]}; tạm bỏ qua 5 phút")

    def report(self):
        for n, s in self.state.items():
            status = s["off"] or ("đang nghỉ" if s["until"] > self.clock() else "sẵn sàng")
            print(f"{n:<11} {s['used']:>5} lượt thành công | {status}")
