"""UI แสดงผล GA จาก ga_grouping.py เป็นกราฟ (ใช้ tkinter ที่มากับ Python ไม่ต้องติดตั้งเพิ่ม)"""
import tkinter as tk
import numpy as np
import ga_grouping as ga

RED, BLUE, GRID, MUTED, FONT = "#e5484d", "#3e63dd", "#e5e7eb", "#6b7280", "Tahoma"


def nice(v):  # ปัดค่าสูงสุดของแกน y ให้เป็นเลขกลม ๆ
    e = 10 ** np.floor(np.log10(v))
    return next(m * e for m in (1, 2, 2.5, 5, 10) if v <= m * e)


def axes(cv, box, ymax, fmt):
    x0, y0, x1, y1 = box
    for i in range(6):
        y = y1 - (y1 - y0) * i / 5
        cv.create_line(x0, y, x1, y, fill=GRID)
        cv.create_text(x0 - 6, y, text=fmt.format(ymax * i / 5), anchor="e", font=(FONT, 8), fill=MUTED)


def bars(cv, box, labels, red, blue, fmt):  # กราฟแท่งคู่ แดง/น้ำเงิน
    x0, y0, x1, y1 = box
    ymax = nice(max(max(red), max(blue)) * 1.15)
    axes(cv, box, ymax, "{:g}")
    slot = (x1 - x0) / len(labels)
    bw = min(slot * 0.35, 44)
    for i, lab in enumerate(labels):
        cx = x0 + slot * (i + 0.5)
        for val, col, dx in ((red[i], RED, -bw), (blue[i], BLUE, 0)):
            top = y1 - (y1 - y0) * val / ymax
            cv.create_rectangle(cx + dx, top, cx + dx + bw, y1, fill=col, outline="")
            cv.create_text(cx + dx + bw / 2, top - 2, text=fmt.format(val), anchor="s", font=(FONT, 8))
        cv.create_text(cx, y1 + 6, text=lab, anchor="n", width=slot, font=(FONT, 9))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HR Spot Day – ผลการแบ่งกลุ่มด้วย GA")
        self.geometry("1100x720")

        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        self.btn = tk.Button(top, text="รัน GA ใหม่", command=self.start)
        self.btn.pack(side="left")
        for name, col in (("■ กลุ่มแดง", RED), ("■ กลุ่มน้ำเงิน", BLUE)):
            tk.Label(top, text=name, fg=col, font=(FONT, 10)).pack(side="left", padx=6)
        self.status = tk.Label(top, font=(FONT, 10))
        self.status.pack(side="left", padx=10)

        grid = tk.Frame(self)
        grid.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.charts = []
        for i, (title, fn) in enumerate([("ค่า cost ที่ดีที่สุดในแต่ละรุ่น", self.draw_cost),
                                         ("จำนวนคนแต่ละแผนก", self.draw_dept),
                                         ("จำนวนชาย / หญิง", self.draw_gender),
                                         ("ค่าเฉลี่ยอายุ / น้ำหนัก / ส่วนสูง", self.draw_mean)]):
            cv = tk.Canvas(grid, bg="white", highlightthickness=0)
            cv.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            cv.bind("<Configure>", lambda e, k=i: self.draw(k))
            self.charts.append((cv, title, fn))
        grid.columnconfigure((0, 1), weight=1)
        grid.rowconfigure((0, 1), weight=1)

        # เปิดหน้าต่างแล้วเริ่มรัน GA ทันที (import ga_grouping ไม่ได้รัน GA ให้ล่วงหน้าแล้ว)
        self.history, self.running, self.gen = [], False, 0
        self.best = ga.random_individual()
        self.after(100, self.start)

    # ---------- รัน GA (ลูปเดียวกับ ga_grouping.py) ทีละ 5 รุ่น ให้หน้าจอไม่ค้าง
    def start(self):
        if self.running:
            return
        self.running, self.gen, self.history = True, 0, []
        self.pop = np.array([ga.random_individual() for _ in range(ga.POP_SIZE)])
        self.btn.config(state="disabled")
        self.step()

    def step(self):
        for _ in range(5):
            c = ga.cost(self.pop)
            self.history.append(c.min())
            self.best = self.pop[np.argmin(c)].copy()
            if self.gen == ga.GENERATIONS:
                self.running = False
                self.btn.config(state="normal")
                break
            self.pop = ga.next_generation(self.pop, c)
            self.gen += 1
        self.redraw()
        if self.running:
            self.after(1, self.step)

    # ---------- วาดกราฟ
    def redraw(self):
        cost = ga.cost(self.best[None])[0]
        self.status.config(text=f"รุ่น {self.gen}/{ga.GENERATIONS}   best cost = {cost:.4f}")
        for k in range(len(self.charts)):
            self.draw(k)

    def draw(self, k):
        cv, title, fn = self.charts[k]
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 120 or h < 120:
            return
        cv.create_text(14, 15, text=title, anchor="w", font=(FONT, 11, "bold"))
        fn(cv, (56, 44, w - 20, h - 44))

    def draw_cost(self, cv, box):
        x0, y0, x1, y1 = box
        if len(self.history) < 2:
            cv.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="กด “รัน GA ใหม่” เพื่อดูกราฟ",
                           font=(FONT, 10), fill=MUTED)
            return
        ymax = nice(max(self.history))
        axes(cv, box, ymax, "{:.2f}")
        pts = []
        for i, v in enumerate(self.history):
            pts += [x0 + (x1 - x0) * i / ga.GENERATIONS, y1 - (y1 - y0) * v / ymax]
        cv.create_line(*pts, fill=BLUE, width=2)
        cv.create_text(x0, y1 + 6, text="0", anchor="n", font=(FONT, 8), fill=MUTED)
        cv.create_text(x1, y1 + 6, text=f"{ga.GENERATIONS} รุ่น", anchor="ne", font=(FONT, 8), fill=MUTED)

    def draw_dept(self, cv, box):
        blue = (self.best @ ga.D).astype(int)
        red = ga.dept_n.astype(int) - blue
        bars(cv, box, ga.depts, red, blue, "{:d}")

    def draw_gender(self, cv, box):
        m = [ga.M[self.best == g] for g in (0, 1)]
        bars(cv, box, ["ชาย", "หญิง"], [int(m[0].sum()), int((1 - m[0]).sum())],
             [int(m[1].sum()), int((1 - m[1]).sum())], "{:d}")

    def draw_mean(self, cv, box):
        red, blue = (ga.X[self.best == g].mean(0) for g in (0, 1))
        bars(cv, box, ["อายุ (ปี)", "น้ำหนัก (กก.)", "ส่วนสูง (ซม.)"], red, blue, "{:.1f}")


if __name__ == "__main__":
    App().mainloop()