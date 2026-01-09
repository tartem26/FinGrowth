import React, { useState, useRef, useLayoutEffect } from "react";
import { useNavigate } from "react-router-dom";
import ReactDOM from "react-dom";
import AppRoutes from "./AppRoutes";
import styles from './styles.js';

// Globals:
// lastCursor: global {x,y} updated on mouse move over the canvas.
// sizeEnum: maps small | medium | ... to [width,height].
// turtle: global mutable object holding drawing state (x,y,angle,penDown,color,lineWidth).
// moveArray: list of movement method names used to generate buttons.
let lastCursor = { x: null, y: null };

const sizeEnum = {
    small: [800, 480],
    medium: [1024, 576],
    large: [1280, 720],
    'absolute unit': [1920, 1080],
};

const turtle = {
    x: 360,
    y: 200,
    angle: 0,
    penDown: true,
    penColor: '#000000',
    lineWidth: 2
};
const moveArray = ['shiftLeft', 'shiftRight', 'shiftUp', 'shiftDown'];

export default function ReactRoot(){
    // State/hooks:
    // contentRef via useRef: points to a scroll container div to measure its layout.
    // isOverflow: boolean used to decide if the scroll container should show a scrollbar and get extra bottom padding.
    // size: selected canvas size string (used to compute width/height).
    // x, y, angle: React state mirrors of turtle.x/y/angle so the turtle marker (triangle <div>) moves/rotates in the React UI.
    // fractal controls:
    //      Koch: kochOrigin (center or cursor), kochOrder
    //      Hilbert: hilbertOrigin, hilbertLevel
    //      hoverMenu: when hover menu is open ('koch' | 'hilbert' | null)

    const contentRef = useRef(null);
    const [isOverflow, setIsOverflow] = useState(false);

    const [size, setSize] = useState('small');

    // turtle position
    const [x, setX] = useState(turtle.x);
    const [y, setY] = useState(turtle.y);
    const [angle, setAngle] = useState(turtle.angle);

    // additional states
    const [kochOrigin, setKochOrigin] = useState('center'); // 'center/ | /cursor'
    const [kochOrder, setKochOrder] = useState(4);

    const [hilbertOrigin, setHilbertOrigin] = useState('center');
    const [hilbertLevel, setHilbertLevel] = useState(5);

    const [hoverMenu, setHoverMenu] = useState(null); // null | 'koch' | 'hilbert'

    // Routing:
    //      useNavigate() from react-router-dom
    //      handleNavigate() calls navigate('/fin-growth-dashboard')
    const navigate = useNavigate();
    const handleNavigate = () => {
        navigate('/fin-growth-dashboard');
    };

    // Drawing triggers:
    //      drawKoch() calls turtle.kochSnowflake(order, {origin, x:lastCursor.x, y:lastCursor.y})
    //      drawHilbert() calls turtle.hilbert(level, {origin, x:lastCursor.x, y:lastCursor.y})
    const drawKoch = (order = kochOrder, origin = kochOrigin) => {
        // clearCanvas();
        turtle.kochSnowflake(Number(order), {
            origin,
            x: lastCursor.x,
            y: lastCursor.y
        });
    };

    const drawHilbert = (level = hilbertLevel, origin = hilbertOrigin) => {
        // clearCanvas();
        turtle.hilbert(Number(level), {
            origin,
            x: lastCursor.x,
            y: lastCursor.y
        });
    };

    // useLayoutEffect runs synchronously after DOM updates, which is useful for
    // layout measurement (scrollHeight/clientHeight) without web flickering.
    // It re-checks overflow when size changes and needs window resize.
    useLayoutEffect(() => {
        const check = () => {
            const el = contentRef.current;
            if (!el) return;
            // true if content needs scrolling
            setIsOverflow(el.scrollHeight > el.clientHeight + 1);
        };

        check();
        window.addEventListener("resize", check);
        return () => window.removeEventListener("resize", check);
    }, [size]);

    setInterval(() => {
        setX(turtle.x);
        setY(turtle.y);
        setAngle(turtle.angle);
    }, 50);


    console.log('turtle X:', turtle.x, ' Y:', turtle.y, ' angle:', turtle.angle );
    const width = sizeEnum[size][0];
    const height = sizeEnum[size][1];
    return (
        <div style={styles.root}>
            <div style={styles.header}>
                <h1 style={styles.ellipseText}>
                    Internship Whitespace
                </h1>
                <div style={styles.stack}>
                    <h4>
                        Canvas Size:
                    </h4>
                    <div style={styles.row}>
                        {Object.keys(sizeEnum).map((key) =>
                            <button
                                key={key}
                                onClick={() => {setSize(key)}}
                                style={{
                                    ...styles.button,
                                    backgroundColor: key === size && '#C9C7C5',
                                    cursor: key !== size && 'pointer',
                                }}
                            >
                                {key}
                            </button>
                        )}
                    </div>
                </div>
            </div>
            <div
                ref={contentRef}
                style={{
                    ...styles.column,
                    overflowY: isOverflow ? "auto" : "hidden",
                    overflowX: "hidden",
                    paddingBottom: isOverflow ? 96 : 36,  // double padding when scrollbar
                }}
            >
                <button
                    onClick={clearCanvas}
                    style={styles.button}
                >
                    Reset Canvas
                </button>
                <div style={{...styles.canvasWrapper, width: width + 2, height: height + 2 }}>
                    <div
                        style={{
                            ...styles.turtle,
                            left: x,
                            top: y,
                            transform: `rotate(${angle}DEG)`,
                        }}
                    />
                    <canvas
                        id="myDrawing"
                        width={width}
                        height={height}
                        onMouseMove={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            lastCursor = {
                                x: e.clientX - rect.left,
                                y: e.clientY - rect.top,
                            };
                        }}
                    />
                </div>
                <h4 style={{ margin: 0 }}>
                    TURTLE FUNCTIONS
                </h4>

                <div style={{ ...styles.row, ...styles.spacer}}>
                    {moveArray.map((key) =>
                        <button
                            key={key}
                            onClick={() => turtle[key]()}
                            style={styles.button}
                        >
                            {key}
                        </button>
                    )}
                </div>

                <div style={{ ...styles.row, maxWidth: width - 48 }}>
                    <button
                        onClick={() => turtle.hexagon()}
                        style={styles.blueButton}
                    >
                        Hexagon
                    </button>
                    <button
                        onClick={() => turtle.drawStar()}
                        style={styles.blueButton}
                    >
                        Star
                    </button>

                    {/* Koch Snowflake with its hover menu */}
                    <div
                        style={{ position: 'relative', display: 'inline-flex' }}
                        onMouseEnter={() => setHoverMenu('koch')}
                        onMouseLeave={() => setHoverMenu(null)}
                    >
                        <button
                            onClick={() => drawKoch()}
                            style={styles.blueButton}
                        >
                            Koch Snowflake
                        </button>

                        {hoverMenu === 'koch' && (
                            <div style={{
                                position: 'absolute',
                                top: 38,
                                left: 0,
                                width: 220,
                                background: 'white',
                                border: '1px solid #000',
                                borderRadius: 10,
                                padding: 10,
                                boxShadow: '0 6px 18px rgba(0,0,0,0.18)',
                                zIndex: 100
                            }}>
                                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Start</div>

                                <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
                                <input
                                    type="radio"
                                    name="kochOrigin"
                                    value="center"
                                    checked={kochOrigin === 'center'}
                                    onChange={() => { setKochOrigin('center'); drawKoch(kochOrder, 'center'); }}
                                />{' '}
                                    Center
                                </label>

                                <label style={{ fontSize: 12, display: 'block', marginBottom: 10 }}>
                                <input
                                    type="radio"
                                    name="kochOrigin"
                                    value="cursor"
                                    checked={kochOrigin === 'cursor'}
                                    onChange={() => { setKochOrigin('cursor'); drawKoch(kochOrder, 'cursor'); }}
                                />{' '}
                                    Cursor
                                </label>

                                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                                    Order: {kochOrder}
                                </div>
                                <input
                                    type="range"
                                    min="1"
                                    max="10"
                                    value={kochOrder}
                                    onChange={(e) => {
                                        const v = Number(e.target.value);
                                        setKochOrder(v);
                                        drawKoch(v, kochOrigin);
                                    }}
                                    onInput={(e) => {
                                        const v = Number(e.target.value);
                                        setKochOrder(v);
                                        drawKoch(v, kochOrigin);
                                    }}
                                    style={{ width: '100%' }}
                                />
                            </div>
                        )}
                    </div>

                    {/* Hilbert with its hover menu */}
                    <div
                        style={{ position: 'relative', display: 'inline-flex' }}
                        onMouseEnter={() => setHoverMenu('hilbert')}
                        onMouseLeave={() => setHoverMenu(null)}
                    >
                        <button
                            onClick={() => drawHilbert()}
                            style={styles.blueButton}
                        >
                            Hilbert
                        </button>

                        {hoverMenu === 'hilbert' && (
                            <div style={{
                                position: 'absolute',
                                top: 38,
                                left: 0,
                                width: 220,
                                background: 'white',
                                border: '1px solid #000',
                                borderRadius: 10,
                                padding: 10,
                                boxShadow: '0 6px 18px rgba(0,0,0,0.18)',
                                zIndex: 100
                            }}>
                                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Start</div>

                                <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
                                <input
                                    type="radio"
                                    name="hilbertOrigin"
                                    value="center"
                                    checked={hilbertOrigin === 'center'}
                                    onChange={() => { setHilbertOrigin('center'); drawHilbert(hilbertLevel, 'center'); }}
                                />{' '}
                                    Center
                                </label>

                                <label style={{ fontSize: 12, display: 'block', marginBottom: 10 }}>
                                <input
                                    type="radio"
                                    name="hilbertOrigin"
                                    value="cursor"
                                    checked={hilbertOrigin === 'cursor'}
                                    onChange={() => { setHilbertOrigin('cursor'); drawHilbert(hilbertLevel, 'cursor'); }}
                                />{' '}
                                    Cursor
                                </label>

                                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                                    Level: {hilbertLevel}
                                </div>
                                <input
                                    type="range"
                                    min="1"
                                    max="10"
                                    value={hilbertLevel}
                                    onChange={(e) => {
                                        const v = Number(e.target.value);
                                        setHilbertLevel(v);
                                        drawHilbert(v, hilbertOrigin);
                                    }}
                                    onInput={(e) => {
                                        const v = Number(e.target.value);
                                        setHilbertLevel(v);
                                        drawHilbert(v, hilbertOrigin);
                                    }}
                                    style={{ width: '100%' }}
                                />
                            </div>
                        )}
                    </div>
                    <button
                        onClick={handleNavigate}
                        style={styles.blueButton}
                    >
                        FinGrowth
                    </button>
                </div>
            </div>
        </div>
    );
}
// react insertion
const wrapper = document.getElementById("react-entry");
wrapper ? ReactDOM.render(<AppRoutes />, wrapper) : false;




// =====================================================================================
//                                  GRAPHICS
// =====================================================================================


// canvas preparation
const canvas = document.getElementById('myDrawing');

if (canvas && canvas.getContext) { // does the browser support 'canvas'?
    turtle.ct = canvas.getContext("2d"); // get drawing context
} else {
    alert('You need a browser which supports the HTML5 canvas!');
}

function clearCanvas () {
    if (canvas && canvas.getContext) {
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);
        turtle.x = 360;
        turtle.y = 200;
    }
}


//      Turtle functions
// =======================================================
turtle.logPenStatus = function () {
    console.log('x=' + this.x + "; y=" + this.y + '; angle = ' + this.angle + '; penDown = ' + this.penDown);
};

// BUG: reposition turtle so you can go off-canvas; fixed below
// turtle.shiftLeft = function (length=50) {
//     turtle.x -= length;
// };
// turtle.shiftRight = function (length=50) {
//     turtle.x += length;
// };
// turtle.shiftUp = function (length=50) {
//     turtle.y -= length;
// };
// turtle.shiftDown = function (length=50) {
//     turtle.y += length;
// };

turtle._clampToCanvas = function (nx, ny) {
    if (!canvas) return { x: nx, y: ny };

    // keep the turtle marker inside the canvas area (only for kochSnowflake
    // and hilbert to compare with hexagon / drawStar)
    const margin = 2; // small margin (set to 0 if need an exact edge)
    const x = Math.max(margin, Math.min(canvas.width - margin, nx));
    const y = Math.max(margin, Math.min(canvas.height - margin, ny));
    return { x, y };
};

// reposition turtle inside the canvas
turtle.shiftLeft = function (length = 50) {
    const p = this._clampToCanvas(this.x - length, this.y);
    this.x = p.x; this.y = p.y;
};

turtle.shiftRight = function (length = 50) {
    const p = this._clampToCanvas(this.x + length, this.y);
    this.x = p.x; this.y = p.y;
};

turtle.shiftUp = function (length = 50) {
    const p = this._clampToCanvas(this.x, this.y - length);
    this.x = p.x; this.y = p.y;
};

turtle.shiftDown = function (length = 50) {
    const p = this._clampToCanvas(this.x, this.y + length);
    this.x = p.x; this.y = p.y;
};

// draw in a direction
turtle.forward = function (length) {
    // this.logPenStatus();
    var x0 = this.x,
        y0 = this.y;
    const angleInRadians = (this.angle * Math.PI) / 180;
    this.x += length * Math.sin(angleInRadians);
    this.y += length * Math.cos(angleInRadians);
    if (this.ct) {
        if (this.penDown) {
            //this.logPenStatus();
            this.ct.beginPath();
            this.ct.lineWidth = this.lineWidth;
            this.ct.strokeStyle = this.penColor;
            this.ct.moveTo(x0, y0);
            this.ct.lineTo(this.x, this.y);
            this.ct.stroke();
        }
    } else {
        this.ct.moveTo(this.x, this.y);
    }
    return this;
};
turtle.backward = function (length) {
    this.forward(-length);
    return this;
};

// turning
turtle.left = function (angle) {
    this.angle += angle;
    return this;
};
turtle.right = function (angle) {
    this.left(-angle);
    return this;
};

// extra turtle utilities

// Core turtle operations
// Move and optionally draw:
//      forward(length) computes a new (x,y) from current angle and draws a line if penDown.
//      It uses:
//          angleInRadians = angle * π / 180
//          x += length * sin(angle)
//          y += length * cos(angle)
// Turn:
//      left(angle) and right(angle) update heading.
// Clamp move buttons:
//      _clampToCanvas(nx, ny) keeps the turtle marker inside the canvas boundaries.
//      shiftLeft/Right/Up/Down update x/y using clamping.
// Utilities used by fractals
//      pushState()/popState(): saves and restores turtle state (position, angle, pen, style).
//      setColor, setLineWidth, moveTo, center, etc.
//      _stopAnim(): cancels any running requestAnimationFrame animation by bumping a token and canceling the prior RAF id.



turtle._stateStack = [];

turtle.pushState = function () {
    this._stateStack.push({
        x: this.x,
        y: this.y,
        angle: this.angle,
        penDown: this.penDown,
        penColor: this.penColor,
        lineWidth: this.lineWidth
    });
    return this;
}

turtle.popState = function () {
    const s = this._stateStack.pop();
    if (!s) return this;
    this.x = s.x;
    this.y = s.y;
    this.angle = s.angle;
    this.penDown = s.penDown;
    this.penColor = s.penColor;
    this.lineWidth = s.lineWidth;
    return this;
};

turtle.penUpFn = function () { this.penDown = false; return this; };
turtle.penDownFn = function () { this.penDown = true; return this; };

turtle.setColor = function (color) { this.penColor = color; return this; };
turtle.setLineWidth = function (w) { this.lineWidth = w; return this; };
turtle.setAngle = function (a) { this.angle = a; return this; };

// move turtle instantly
turtle.moveTo = function (x, y, draw = false) {
    if (this.ct && draw && this.penDown) {
        this.ct.beginPath();
        this.ct.lineWidth = this.lineWidth;
        this.ct.strokeStyle = this.penColor;
        this.ct.moveTo(this.x, this.y);
        this.ct.lineTo(x, y);
        this.ct.stroke();
    }
    this.x = x;
    this.y = y;
    return this;
};

turtle.center = function () {
    if (typeof canvas !== "undefined" && canvas) {
        this.x = canvas.width / 2;
        this.y = canvas.height / 2;
    }
    return this;
};

turtle._anim = {token: 0, rafId: null};

turtle._stopAnim = function () {
    this._anim.token += 1;
    if (this._anim.rafId) {
        cancelAnimationFrame(this._anim.rafId);
        this._anim.rafId = null;
    }
};




// ===============================================================
//                      Pattern Functions
// ===============================================================

turtle.hexagon = function (length=50) {
    console.log('length', length);
    var i;
    for (i = 1; i <= 6; i++) {
        turtle.forward(length);
        turtle.left(60);

    }
};

turtle.drawStar = function () {
    var i;
    for (i = 0; i < 18; i++) {
        turtle.left(100);
        turtle.forward(80);
    }
};

// Draws 3 Koch curves (triangle sides).
// Each koch(lvl, len) expands into 4 smaller segments with turns in between:
//      A, L60, B, R120, C, L60, D
// Uses stack.push(...) in reverse order so popping executes in correct order.
turtle.kochSnowflake = function (order = 4, opts = {}) {
    if (typeof canvas === "undefined" || !canvas) return;
    this._stopAnim();

    const n = Math.max(1, Math.min(10, Math.floor(order)));

    // origin (center or cursor)
    let ox = canvas.width / 2;
    let oy = canvas.height / 2;
    if (opts.origin === 'cursor' && Number.isFinite(this.x) && Number.isFinite(this.y)) {
        ox = this.x;
        oy = this.y;
    }

    // Koch outward padding factor (max bump distance)
    const sqrt3 = Math.sqrt(3);
    const k = (sqrt3 / 12) * (1 - Math.pow(1 / 3, n)); // pad = side * k

    // choose side length so full snowflake fits
    const sideMaxW = canvas.width / (1 + 2 * k);
    const sideMaxH = canvas.height / (sqrt3 / 2 + 2 * k);
    const side = Math.min(sideMaxW, sideMaxH) * 0.92;

    const pad = side * k;
    const W = side * (1 + 2 * k);
    const H = side * (sqrt3 / 2 + 2 * k);

    // clamp origin so bounding box stays inside canvas
    ox = Math.max(W / 2, Math.min(canvas.width - W / 2, ox));
    oy = Math.max(H / 2, Math.min(canvas.height - H / 2, oy));

    // start point: bottom-left inside padded bounding box
    const startX = ox - W / 2 + pad;
    const startY = oy + H / 2 - pad;

    // save and setup
    const saved = { x: this.x, y: this.y, angle: this.angle, penDown: this.penDown, penColor: this.penColor, lineWidth: this.lineWidth };
    this.penDownFn();
    this.setLineWidth(1.5);
    this.setAngle(90);
    this.moveTo(startX, startY, false);

    const totalSeg = 3 * Math.pow(4, n);
    let seg = 0;

    const stepColor = () => {
        const t = totalSeg <= 1 ? 0 : seg / (totalSeg - 1);
        const hue = 200 + 140 * t; // blue -> purple
        this.setColor(`hsl(${hue}, 80%, 35%)`);
    };

    // iterative stack to avoid recursion and allow animation
    const stack = [];
    // need at least for 3 sides; thus, koch(n, side), then left(120)
    for (let i = 0; i < 3; i++) {
        stack.push({ type: 'turnL', a: 120 });
        stack.push({ type: 'koch', lvl: n, len: side });
    }

    const token = this._anim.token;
    const opsPerFrame = 4000;

    const tick = () => {
        if (token !== this._anim.token) return;

        let ops = 0;
        while (ops < opsPerFrame && stack.length) {
            const job = stack.pop();

            if (job.type === 'turnL') {
                this.left(job.a);
                ops++;
                continue;
            }
            if (job.type === 'turnR') {
                this.right(job.a);
                ops++;
                continue;
            }
            if (job.type === 'fwd') {
                stepColor();
                this.forward(job.len);
                seg++;
                ops++;
                continue;
            }
            if (job.type === 'koch') {
                const lvl = job.lvl;
                const len = job.len;
                if (lvl === 0) {
                    stack.push({ type: 'fwd', len });
                } else {
                    const l = len / 3;
                    // reverse push of:
                    // A, L60, B, R120, C, L60, D
                    stack.push({ type: 'koch', lvl: lvl - 1, len: l }); // D
                    stack.push({ type: 'turnL', a: 60 });
                    stack.push({ type: 'koch', lvl: lvl - 1, len: l }); // C
                    stack.push({ type: 'turnR', a: 120 });
                    stack.push({ type: 'koch', lvl: lvl - 1, len: l }); // B
                    stack.push({ type: 'turnL', a: 60 });
                    stack.push({ type: 'koch', lvl: lvl - 1, len: l }); // A
                }
                continue;
            }
        }

        if (stack.length) {
            this._anim.rafId = requestAnimationFrame(tick);
        } else {
            // restore
            this.x = saved.x; this.y = saved.y; this.angle = saved.angle;
            this.penDown = saved.penDown; this.penColor = saved.penColor; this.lineWidth = saved.lineWidth;
            this._anim.rafId = null;
        }
    };

    this._anim.rafId = requestAnimationFrame(tick);
};

// Uses standard Hilbert L-system–like expansion pattern.
// Computes:
//      step = size / (2^n - 1)
//      total segments = 4^n - 1
// Expands jobs like:
//      turn, recurse, forward, turn, recurse, forward, recurse, turn, forward, recurse, turn (etc.)
// Also stack-based and animated.
turtle.hilbert = function (level = 5, opts = {}) {
    if (typeof canvas === "undefined" || !canvas) return;
    this._stopAnim();

    const n = Math.max(1, Math.min(10, Math.floor(level)));

    // origin (center or cursor)
    let ox = canvas.width / 2;
    let oy = canvas.height / 2;
    if (opts.origin === 'cursor' && Number.isFinite(this.x) && Number.isFinite(this.y)) {
        ox = this.x;
        oy = this.y;
    }

    const size = Math.min(canvas.width, canvas.height) * 0.78;

    // clamp so square stays visible
    ox = Math.max(size / 2, Math.min(canvas.width - size / 2, ox));
    oy = Math.max(size / 2, Math.min(canvas.height - size / 2, oy));

    const startX = ox - size / 2;
    const startY = oy - size / 2;

    // step size depends on level
    const step = size / (Math.pow(2, n) - 1);

    const saved = { x: this.x, y: this.y, angle: this.angle, penDown: this.penDown, penColor: this.penColor, lineWidth: this.lineWidth };
    this.penDownFn();
    this.setLineWidth(1.2);
    this.setAngle(90);
    this.moveTo(startX, startY, false);

    let i = 0;
    const total = Math.pow(4, n) - 1;

    const stepColor = () => {
        const t = total <= 1 ? 0 : i / (total - 1);
        const hue = 20 + 300 * t;
        this.setColor(`hsl(${hue}, 85%, 40%)`);
    };

    // iterative Hilbert expansion
    const stack = [{ type: 'hil', lvl: n, ang: 90 }];

    const token = this._anim.token;
    const opsPerFrame = 5000;

    const tick = () => {
        if (token !== this._anim.token) return;

        let ops = 0;
        while (ops < opsPerFrame && stack.length) {
            const job = stack.pop();

            if (job.type === 'turnL') { this.left(job.a); ops++; continue; }
            if (job.type === 'turnR') { this.right(job.a); ops++; continue; }
            if (job.type === 'fwd') {
                stepColor();
                this.forward(step);
                i++;
                ops++;
                continue;
            }

            if (job.type === 'hil') {
                const lvl = job.lvl;
                const ang = job.ang;
                if (lvl === 0) continue;

                stack.push({ type: 'turnR', a: ang });
                stack.push({ type: 'hil', lvl: lvl - 1, ang: -ang });
                stack.push({ type: 'fwd' });
                stack.push({ type: 'turnL', a: ang });
                stack.push({ type: 'hil', lvl: lvl - 1, ang: ang });
                stack.push({ type: 'fwd' });
                stack.push({ type: 'hil', lvl: lvl - 1, ang: ang });
                stack.push({ type: 'turnL', a: ang });
                stack.push({ type: 'fwd' });
                stack.push({ type: 'hil', lvl: lvl - 1, ang: -ang });
                stack.push({ type: 'turnR', a: ang });

                continue;
            }
        }

        if (stack.length) {
            this._anim.rafId = requestAnimationFrame(tick);
        } else {
            this.x = saved.x; this.y = saved.y; this.angle = saved.angle;
            this.penDown = saved.penDown; this.penColor = saved.penColor; this.lineWidth = saved.lineWidth;
            this._anim.rafId = null;
        }
    };

    this._anim.rafId = requestAnimationFrame(tick);
};

