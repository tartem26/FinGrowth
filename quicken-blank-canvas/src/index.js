import React, { useState } from "react";
import ReactDOM from "react-dom";
import styles from './styles.js';


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

function ReactRoot(){
    const [size, setSize] = useState('small');

    // turtle position
    const [x, setX] = useState(turtle.x);
    const [y, setY] = useState(turtle.y);
    const [angle, setAngle] = useState(turtle.angle);

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
            <div style={styles.column}>
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
                    <button
                        onClick={() => turtle.kochSnowflake()}
                        style={styles.blueButton}
                    >
                        Koch Snowflake
                    </button>
                    <button
                        onClick={() => turtle.hilbert()}
                        style={styles.blueButton}
                    >
                        Hilbert
                    </button>
                    <button
                        onClick={() => console.log('yo')}
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
wrapper ? ReactDOM.render(<ReactRoot />, wrapper) : false;




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

// reposition turtle
turtle.shiftLeft = function (length=50) {
    turtle.x -= length;
};
turtle.shiftRight = function (length=50) {
    turtle.x += length;
};
turtle.shiftUp = function (length=50) {
    turtle.y -= length;
};
turtle.shiftDown = function (length=50) {
    turtle.y += length;
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

turtle.kochSnowflake = function (order = 4) {
    if (typeof canvas === "undefined" || !canvas) return;

    this.pushState();
    this.penDownFn();
    this.setLineWidth(1.5);

    const side = Math.min(canvas.width, canvas.height) * 0.62;
    const h = side * Math.sqrt(3) / 2;

    const startX = (canvas.width - side) / 2;
    const startY = (canvas.height + h) / 2;

    this.setAngle(90);
    this.moveTo(startX, startY, false);

    // segments of order of 4
    let seg = 0;
    const totalSeg = 3 * Math.pow(4, order);
    const colorize = true;

    const stepColor = () => {
        if (!colorize) return;
        const t = totalSeg <= 1 ? 0 : seg / (totalSeg - 1);
        const hue = 200 + 140 * t; // blue -> purple
        this.setColor(`hsl(${hue}, 80%, 35%)`);
    };

    const koch = (n, len) => {
        if (n === 0) {
            stepColor();
            this.forward(len);
            seg++;
            return;
        }
        len /= 3;
        koch(n - 1, len);
        this.left(60);
        koch(n - 1, len);
        this.right(120);
        koch(n - 1, len);
        this.left(60);
        koch(n - 1, len);
    };

    for (let i = 0; i < 3; i++) {
        koch(order, side);
        this.right(120);
    }

    this.popState();
};

turtle.hilbert = function (level = 5) {
    if (typeof canvas === "undefined" || !canvas) return;

    this.pushState();
    this.penDownFn();
    this.setLineWidth(1.2);

    const n = level;

    const size = Math.min(canvas.width, canvas.height) * 0.78;
    const step = size / (Math.pow(2, n) - 1);

    const startX = (canvas.width - size) / 2;
    const startY = (canvas.height - size) / 2;

    this.setAngle(90);          // start facing right
    this.moveTo(startX, startY, false);

    // Hilbert curve segments: 4^n - 1
    let i = 0;
    const total = Math.pow(4, n) - 1;
    const colorize = true;

    const stepColor = () => {
        if (!colorize) return;
        const t = total <= 1 ? 0 : i / (total - 1);
        const hue = 20 + 300 * t; // warm -> rainbow
        this.setColor(`hsl(${hue}, 85%, 40%)`);
    };

    const hilbert = (lvl, angle) => {
        if (lvl === 0) return;

        this.right(angle);
        hilbert(lvl - 1, -angle);

        stepColor(); this.forward(step); i++;

        this.left(angle);
        hilbert(lvl - 1, angle);

        stepColor(); this.forward(step); i++;

        hilbert(lvl - 1, angle);

        this.left(angle);
        stepColor(); this.forward(step); i++;

        hilbert(lvl - 1, -angle);
        this.right(angle);
    };

    hilbert(n, 90);

    this.popState();
};