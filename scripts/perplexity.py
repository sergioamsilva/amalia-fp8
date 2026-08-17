"""Measure perplexity over European Portuguese text.

This is the correct metric for judging a quantization. Unlike token-by-token agreement, it
does not suffer from the greedy-decoding cascade: it measures the model's confidence in the
SAME tokens rather than in the path it would choose.

Usage:
    TEST_URL=http://host:8000/v1/completions OUTPUT=results/ppl_fp8.json \\
    python3 scripts/perplexity.py
"""
import json, math, os, urllib.request

URL = os.environ.get("TEST_URL", "http://localhost:8000/v1/completions")
MODEL = os.environ.get("TEST_MODEL", "amalia")
OUTPUT = os.environ.get("OUTPUT", "results/ppl.json")

# NOTE: the passages below are in Portuguese on purpose. Perplexity has to be measured
# on the language the model is specialised in.
# Four registers of European Portuguese. The colloquial passage is deliberately dense with
# comboio, pequeno-almoço, casa de banho and telemóvel, so that any degradation of
# specifically European vocabulary shows up there.
TEXTS = {
    "literary": (
        "Num lugar da Mancha, de cujo nome não quero lembrar-me, vivia não há muito tempo "
        "um fidalgo dos de lança em cabido, adarga antiga, rocim magro e galgo corredor. "
        "A sua casa era de telha vã, com paredes caiadas e um alpendre onde a videira dava "
        "sombra no verão. Todas as manhãs descia à vila para comprar o pão e o jornal, "
        "cumprimentava os vizinhos pelo nome e regressava devagar, pela estrada de terra "
        "batida, com o cão a seguir-lhe os passos."
    ),
    "administrative": (
        "Nos termos do disposto no artigo vigésimo terceiro do Regulamento Geral de Proteção "
        "de Dados, o titular dos dados tem o direito de solicitar ao responsável pelo "
        "tratamento o acesso aos dados pessoais que lhe digam respeito, bem como a sua "
        "retificação ou o seu apagamento. O pedido deve ser dirigido ao encarregado de "
        "proteção de dados, por escrito, e respondido no prazo máximo de um mês a contar "
        "da data da sua receção."
    ),
    "colloquial": (
        "Ontem apanhei o comboio das oito para o Porto e cheguei mesmo à hora do "
        "pequeno-almoço. Deixei a mala no hotel, perguntei onde ficava a casa de banho, "
        "e saí à rua com o telemóvel na mão à procura de um sítio para tomar café. "
        "Estava a chover à bruxa, mas nem por isso deixei de dar uma volta pela Ribeira "
        "antes da reunião."
    ),
    "technical": (
        "A travessia de diretórios ocorre quando uma aplicação constrói um caminho de "
        "ficheiro a partir de dados fornecidos pelo utilizador sem validar o resultado. "
        "Um atacante que consiga introduzir sequências de subida na hierarquia consegue "
        "ler ficheiros fora da raiz permitida. A mitigação correta consiste em normalizar "
        "o caminho e verificar que o resultado continua contido na diretoria base, e nunca "
        "em filtrar sequências de caracteres suspeitas."
    ),
}


def perplexity(text):
    body = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 0, "echo": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    choice = d["choices"][0]
    prompt_logprobs = choice.get("prompt_logprobs") or []
    logprobs = []
    for entry in prompt_logprobs:
        if not entry:
            continue  # the first token has no logprob
        # With prompt_logprobs=0 vLLM returns exactly ONE entry per position, and that
        # entry belongs to the token actually present in the text (its rank can be
        # anything, it is not 1). If more entries arrive it is because prompt_logprobs>0,
        # and then there is no safe way to identify the real token: fail loudly rather
        # than return a number that is wrong and flatteringly low.
        if len(entry) != 1:
            raise RuntimeError(
                f"expected 1 logprob per position (prompt_logprobs=0), got {len(entry)}. "
                "Do not use prompt_logprobs>0 with this script."
            )
        (info,) = entry.values()
        logprobs.append(info["logprob"])
    if not logprobs:
        raise RuntimeError("the server returned no prompt_logprobs")
    nll = -sum(logprobs) / len(logprobs)
    return math.exp(nll), len(logprobs)


print(f"model={MODEL} @ {URL}")
print("=" * 62)
results = {}
total_nll = 0.0
total_n = 0
for name, text in TEXTS.items():
    try:
        ppl, n = perplexity(text)
    except Exception as e:
        print(f"  {name:16s} ERROR {type(e).__name__}: {e}")
        continue
    results[name] = {"ppl": round(ppl, 4), "tokens": n}
    total_nll += math.log(ppl) * n
    total_n += n
    print(f"  {name:16s} ppl={ppl:8.4f}  ({n} tokens)")
global_ppl = math.exp(total_nll / total_n) if total_n else 0
print("=" * 62)
print(f"  global perplexity: {global_ppl:.4f}  ({total_n} tokens)")
results["_global"] = {"ppl": round(global_ppl, 4), "tokens": total_n}

os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"  written to {OUTPUT}")
