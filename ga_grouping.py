"""Genetic Algorithm แบ่งพนักงานเป็น 2 กลุ่ม (แดง/น้ำเงิน) ให้ยุติธรรมและได้รู้จักข้ามแผนก"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent                     # อ้างอิงไฟล์จากโฟลเดอร์ของสคริปต์ รันจากที่ไหนก็ได้
INPUT = HERE / "Data_sheet" / "HR(in).csv"
OUTPUT = HERE / "HR_groups.csv"
POP_SIZE, GENERATIONS, ELITE = 300, 800, 6
SEED = 42
rng = np.random.default_rng(SEED)

# ---------- โหลดข้อมูล
with open(INPUT, newline="", encoding="utf-8-sig") as f:
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f) if r["gender"]]
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
        w = csv.writer(f)
        w.writerow(["gender", "department", "weight", "height", "age", "group"])
        for r, g in zip(rows, best):
            w.writerow([r["gender"], r["department"], r["weight"], r["height"], r["age"], ["Red", "Blue"][g]])
    print(f"บันทึก {Path(path).name} แล้ว")


def main():
    best = run_ga()
    summary(best)
    save(best)


# รัน GA เฉพาะตอนสั่ง python ga_grouping.py ตรง ๆ (import จาก UI.py จะไม่รันซ้ำ และไม่เขียนทับไฟล์ผล)
if __name__ == "__main__":
    main()
