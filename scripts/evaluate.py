"""Evaluate one AMALIA variant and store its outputs for comparison.

Usage:
    TEST_URL=http://host:8000/v1/chat/completions TEST_MODEL=amalia \\
    OUTPUT=results/bf16.json python3 scripts/evaluate.py

The point is to run the same prompt set with deterministic generation (temperature 0)
against every variant, then compare the outputs against the BF16 reference.

Two things are measured:
  1. exact agreement with the reference (see the caveat in the README: this metric is
     unsuitable for judging quantization quality, use perplexity.py instead)
  2. European vs Brazilian Portuguese lexical markers
"""
import json, os, re, time, urllib.request

URL = os.environ.get("TEST_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("TEST_MODEL", "amalia")
OUTPUT = os.environ.get("OUTPUT", "results/output.json")

# NOTE: the prompts below are in Portuguese on purpose. The model under test is a
# European Portuguese model, so the evaluation corpus has to be Portuguese. Everything
# else in this repository is English.
# Prompts chosen to exercise vocabulary, grammar and register in European Portuguese,
# plus a few general-knowledge ones so the set does not only measure dialect.
PROMPTS = [
    ("transport", "Descreve em duas frases uma viagem de Lisboa ao Porto de transporte ferroviario."),
    ("house", "Estou num hotel e preciso de indicar onde fica a divisao com o duche. Escreve duas frases."),
    ("mobile", "Escreve duas frases sobre como as pessoas usam o telefone portatil no dia a dia."),
    ("gerund", "Descreve o que estas a fazer neste momento, em duas frases."),
    ("breakfast", "Descreve a primeira refeicao do dia em Portugal, em duas frases."),
    ("bus", "Explica em duas frases como se apanha transporte publico rodoviario numa cidade."),
    ("formal_register", "Escreve um paragrafo formal a informar um cliente de que a fatura vence a 30 dias."),
    ("history", "Quem foi Luis de Camoes e porque e importante? Responde em tres frases."),
    ("geography", "Quais sao os arquipelagos portugueses e onde ficam? Responde em duas frases."),
    ("technical", "Explica em tres frases o que e uma vulnerabilidade de path traversal."),
    ("reasoning", "Se um comboio parte as 9h15 e demora 2h50, a que horas chega? Explica o calculo."),
    ("code", "Escreve uma funcao Python que devolve os numeros primos ate n. So o codigo."),
]

# Lexical markers: (European Portuguese, Brazilian Portuguese).
# The prompts stay in Portuguese on purpose, since the model under test is a Portuguese one.
MARKERS = [
    ("comboio", "trem"),                  # train
    ("casa de banho", "banheiro"),        # bathroom
    ("telemovel", "celular"),             # mobile phone
    ("pequeno-almoco", "cafe da manha"),  # breakfast
    ("autocarro", "onibus"),              # bus
    ("ecra", "tela"),                     # screen
    ("frigorifico", "geladeira"),         # fridge
    ("sandes", "sanduiche"),              # sandwich
    ("bilhete", "passagem"),              # ticket
    ("utilizador", "usuario"),            # user
    ("ficheiro", "arquivo"),              # file
]


def strip_accents(s):
    s = s.lower()
    for a, b in [("á", "a"), ("à", "a"), ("â", "a"), ("ã", "a"), ("é", "e"), ("ê", "e"),
                 ("í", "i"), ("ó", "o"), ("ô", "o"), ("õ", "o"), ("ú", "u"), ("ç", "c")]:
        s = s.replace(a, b)
    return s


def dialect_markers(text):
    """Count markers of each variety and detect the Brazilian gerund construction."""
    t = strip_accents(text)
    pt = sum(1 for a, _ in MARKERS if a in t)
    br = sum(1 for _, b in MARKERS if b in t)
    # "estou fazendo" (BR) vs "estou a fazer" (PT)
    ger_br = len(re.findall(r"\b(estou|estas|esta|estamos|estao)\s+\w+ndo\b", t))
    ger_pt = len(re.findall(r"\b(estou|estas|esta|estamos|estao)\s+a\s+\w+r\b", t))
    return pt, br, ger_br, ger_pt


def ask(prompt, max_tokens=350):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 42}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), d["usage"], dt


answers = {}
tot_pt = tot_br = tot_gbr = tot_gpt = 0
tot_tokens = 0.0
tot_time = 0.0
print(f"model={MODEL} @ {URL}")
print("=" * 78)
for name, prompt in PROMPTS:
    try:
        text, usage, dt = ask(prompt)
    except Exception as e:
        print(f"  {name:16s} ERROR {type(e).__name__}: {e}")
        answers[name] = {"error": str(e)}
        continue
    pt, br, gbr, gpt = dialect_markers(text)
    tot_pt += pt; tot_br += br; tot_gbr += gbr; tot_gpt += gpt
    n = usage["completion_tokens"]; tot_tokens += n; tot_time += dt
    answers[name] = {"prompt": prompt, "answer": text, "tokens": n, "seconds": round(dt, 2)}
    print(f"  {name:16s} {n:4d} tok {dt:6.1f}s  PT={pt} BR={br} ger_pt={gpt} ger_br={gbr}")

summary = {
    "model": MODEL, "url": URL,
    "markers_pt": tot_pt, "markers_br": tot_br,
    "gerund_pt": tot_gpt, "gerund_br": tot_gbr,
    "tokens_total": tot_tokens, "seconds_total": round(tot_time, 1),
    "tokens_per_s": round(tot_tokens / tot_time, 1) if tot_time else 0,
}
print("=" * 78)
print(f"  European Portuguese markers: {tot_pt} | Brazilian: {tot_br}")
print(f"  gerund PT (estou a fazer): {tot_gpt} | BR (estou fazendo): {tot_gbr}")
print(f"  mean throughput: {summary['tokens_per_s']} tok/s")

os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump({"summary": summary, "answers": answers}, f, ensure_ascii=False, indent=2)
print(f"  written to {OUTPUT}")
