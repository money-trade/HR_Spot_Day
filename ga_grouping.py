"""Genetic Algorithm แบ่งพนักงานเป็น 2 กลุ่ม (แดง/น้ำเงิน) ให้ยุติธรรมและได้รู้จักข้ามแผนก"""
import csv
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent                     # อ้างอิงไฟล์จากโฟลเดอร์ของสคริปต์ รันจากที่ไหนก็ได้
INPUT = HERE / "Data_sheet" / "HR(in).csv"
OUTPUT = HERE / "HR_groups.csv"
POP_SIZE, GENERATIONS, ELITE = 300, 800, 6
SEED = 42
rng = np.random.default_rng(SEED)

# ---------- โหลดข้อมูล (ตรวจความถูกต้องก่อน แถวที่มีปัญหาจะถูกข้ามและแจ้งเลขแถว)
REQUIRED = ("gender", "department", "weight", "height", "age")
GENDERS = ("Male", "Female")


def load_rows(path):
    if not path.exists():
        sys.exit(f"ไม่พบไฟล์ข้อมูล: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in {h.strip() for h in reader.fieldnames or []}]
        if missing:
            sys.exit(f"ไฟล์ {path.name} ไม่มีคอลัมน์: {', '.join(missing)}")
        rows, skipped = [], []
        for line, raw in enumerate(reader, start=2):  # แถวที่ 1 คือหัวตาราง
            r = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            if not any(r.values()):
                continue  # แถวว่าง
            if r["gender"] not in GENDERS:
                skipped.append(f"แถว {line}: gender = '{r['gender']}' (ต้องเป็น Male หรือ Female)")
            elif not r["department"]:
                skipped.append(f"แถว {line}: ไม่มี department")
            else:
                try:
                    [float(r[c]) for c in ("age", "weight", "height")]
                    rows.append(r)
                except ValueError:
                    skipped.append(f"แถว {line}: อายุ/น้ำหนัก/ส่วนสูง ไม่ใช่ตัวเลข")
    if skipped:
        print(f"ข้าม {len(skipped)} แถวที่ข้อมูลไม่ครบหรือผิดรูปแบบ:")
        for s in skipped:
            print("  " + s)
    if len(rows) % 2:
        print(f"เตือน: มี {len(rows)} คน (เลขคี่) กลุ่มแดงจะมากกว่ากลุ่มน้ำเงิน 1 คน")
    return rows


rows = load_rows(INPUT)
N, HALF = len(rows), len(rows) // 2
depts = sorted({r["department"] for r in rows})
D = np.array([[r["department"] == d for d in depts] for r in rows], float)  # one-hot แผนก
M = np.array([r["gender"] == "Male" for r in rows], float)                   # 1 = ชาย
X = np.array([[float(r[c]) for c in ("age", "weight", "height")] for r in rows])
dept_n = D.sum(0)


# ---------- fitness: ความไม่สมดุล (ยิ่งต่ำยิ่งดี)  pop = P x N, 1 = น้ำเงิน, 0 = แดง
def cost(pop):
    inv = 1 - pop
    d1 = pop @ D
    dept = (np.abs(2 * d1 - dept_n) - dept_n % 2).sum(1) / N        # แต่ละแผนกแบ่งครึ่ง
    gender = np.abs(pop @ M - inv @ M) / M.sum()                     # จำนวนชาย/หญิงเท่ากัน
    mean = (np.abs(pop @ X - inv @ X) / HALF / X.std(0)).sum(1)       # ค่าเฉลี่ยอายุ/น้ำหนัก/ส่วนสูง
    return 3 * dept + 2 * gender + mean


# ---------- operators
def random_individual():
    ind = np.zeros(N, np.int8)
    ind[rng.choice(N, HALF, replace=False)] = 1
    return ind


def crossover(a, b):  # uniform crossover + ซ่อมให้ 2 กลุ่มมีขนาดเท่ากัน
    child = np.where(rng.random(N) < 0.5, a, b).astype(np.int8)
    diff = int(child.sum()) - HALF
    side = 1 if diff > 0 else 0
    child[rng.choice(np.flatnonzero(child == side), abs(diff), replace=False)] = 1 - side
    return child


def mutate(ind):  # สลับคนแดง <-> น้ำเงิน 1-3 คู่
    for _ in range(rng.integers(1, 4)):
        ind[rng.choice(np.flatnonzero(ind == 1))] = 0
        ind[rng.choice(np.flatnonzero(ind == 0))] = 1
    return ind


def pick(pop, c):  # tournament selection
    i = rng.integers(0, len(pop), 4)
    return pop[i[np.argmin(c[i])]]


def local_search(ind):
    """หลัง GA: ลองสลับคนแดง 1 คนกับน้ำเงิน 1 คนทุกคู่ (179 x 179 แบบ) เลือกคู่ที่ cost ลดมากสุด
    ทำซ้ำจนสลับคู่ไหนก็ไม่ดีขึ้น (hill climbing) GA หาบริเวณที่ดี ส่วนนี้เก็บรายละเอียดสุดท้าย"""
    ind = ind.copy()
    cur = cost(ind[None])[0]
    while True:
        blue, red = np.flatnonzero(ind == 1), np.flatnonzero(ind == 0)
        b, r = np.repeat(blue, len(red)), np.tile(red, len(blue))
        cand = np.repeat(ind[None], len(b), 0)
        k = np.arange(len(b))
        cand[k, b], cand[k, r] = 0, 1
        c = cost(cand)
        j = int(np.argmin(c))
        if c[j] >= cur - 1e-12:
            return ind
        ind, cur = cand[j], c[j]


def next_generation(pop, c):  # elitism + ลูกจาก crossover/mutation (UI.py ใช้ฟังก์ชันนี้ด้วย)
    elite = pop[np.argsort(c)[:ELITE]]
    children = [mutate(crossover(pick(pop, c), pick(pop, c))) for _ in range(POP_SIZE - ELITE)]
    return np.vstack([elite, children])


# ---------- วน GA
def run_ga(log=True):
    pop = np.array([random_individual() for _ in range(POP_SIZE)])
    for gen in range(GENERATIONS):
        c = cost(pop)
        if log and gen % 100 == 0:
            print(f"gen {gen:4d}  best cost = {c.min():.4f}")
        pop = next_generation(pop, c)
    return pop[np.argmin(cost(pop))]


# ---------- สรุปผล และบันทึกไฟล์
def summary(best):
    print(f"\nbest cost = {cost(best[None])[0]:.4f}")
    for g, name in enumerate(["Red", "Blue"]):
        m = best == g
        print(f"{name:<5} n={m.sum()}  ชาย={int(M[m].sum())} หญิง={int((1 - M[m]).sum())}  "
              f"อายุ={X[m, 0].mean():.1f} น้ำหนัก={X[m, 1].mean():.1f} ส่วนสูง={X[m, 2].mean():.1f}")
    d1 = (best @ D).astype(int)
    print("แผนก (แดง/น้ำเงิน): " + "  ".join(f"{d}:{int(n) - b}/{b}" for d, n, b in zip(depts, dept_n, d1)))


def save(best, path=OUTPUT):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["gender", "department", "weight", "height", "age", "group"])
        for r, g in zip(rows, best):
            w.writerow([r["gender"], r["department"], r["weight"], r["height"], r["age"], ["Red", "Blue"][g]])
    print(f"บันทึก {Path(path).name} แล้ว")


def load_groups(path=OUTPUT):
    """อ่านผลเดิมจาก HR_groups.csv กลับมาเป็นโครโมโซม (None ถ้าไม่มีไฟล์หรือข้อมูลไม่ตรงกับ HR(in).csv)"""
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        saved = list(csv.DictReader(f))
    if len(saved) != N:
        return None
    ind = np.zeros(N, np.int8)
    for i, (r, s) in enumerate(zip(rows, saved)):
        if any(r[c] != s.get(c, "").strip() for c in REQUIRED) or s.get("group") not in ("Red", "Blue"):
            return None
        ind[i] = s["group"] == "Blue"
    return ind


def main():
    best = run_ga()
    print(f"\nGA จบที่ cost = {cost(best[None])[0]:.4f}  -> ปรับต่อด้วย local search")
    best = local_search(best)
    summary(best)
    save(best)


# รัน GA เฉพาะตอนสั่ง python ga_grouping.py ตรง ๆ (import จาก UI.py จะไม่รันซ้ำ และไม่เขียนทับไฟล์ผล)
if __name__ == "__main__":
    main()
