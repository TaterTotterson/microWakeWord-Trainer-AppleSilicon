//#region node_modules/@vue/shared/dist/shared.esm-bundler.js
// @__NO_SIDE_EFFECTS__
function e(e) {
	let t = /* @__PURE__ */ Object.create(null);
	for (let n of e.split(",")) t[n] = 1;
	return (e) => e in t;
}
var t = {}, n = [], r = () => {}, i = () => !1, a = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && (e.charCodeAt(2) > 122 || e.charCodeAt(2) < 97), o = (e) => e.startsWith("onUpdate:"), s = Object.assign, c = (e, t) => {
	let n = e.indexOf(t);
	n > -1 && e.splice(n, 1);
}, l = Object.prototype.hasOwnProperty, u = (e, t) => l.call(e, t), d = Array.isArray, f = (e) => x(e) === "[object Map]", p = (e) => x(e) === "[object Set]", m = (e) => x(e) === "[object Date]", h = (e) => typeof e == "function", g = (e) => typeof e == "string", _ = (e) => typeof e == "symbol", v = (e) => typeof e == "object" && !!e, y = (e) => (v(e) || h(e)) && h(e.then) && h(e.catch), b = Object.prototype.toString, x = (e) => b.call(e), S = (e) => x(e).slice(8, -1), C = (e) => x(e) === "[object Object]", w = (e) => g(e) && e !== "NaN" && e[0] !== "-" && "" + parseInt(e, 10) === e, ee = /* @__PURE__ */ e(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"), te = (e) => {
	let t = /* @__PURE__ */ Object.create(null);
	return ((n) => t[n] || (t[n] = e(n)));
}, ne = /-\w/g, T = te((e) => e.replace(ne, (e) => e.slice(1).toUpperCase())), re = /\B([A-Z])/g, E = te((e) => e.replace(re, "-$1").toLowerCase()), ie = te((e) => e.charAt(0).toUpperCase() + e.slice(1)), ae = te((e) => e ? `on${ie(e)}` : ""), oe = (e, t) => !Object.is(e, t), se = (e, ...t) => {
	for (let n = 0; n < e.length; n++) e[n](...t);
}, D = (e, t, n, r = !1) => {
	Object.defineProperty(e, t, {
		configurable: !0,
		enumerable: !1,
		writable: r,
		value: n
	});
}, ce = (e) => {
	let t = parseFloat(e);
	return isNaN(t) ? e : t;
}, le = (e) => {
	let t = g(e) ? Number(e) : NaN;
	return isNaN(t) ? e : t;
}, ue, de = () => ue ||= typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {};
function fe(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) {
			let r = e[n], i = g(r) ? ge(r) : fe(r);
			if (i) for (let e in i) t[e] = i[e];
		}
		return t;
	}
	if (g(e) || v(e)) return e;
}
var pe = /;(?![^(]*\))/g, me = /:([^]+)/, he = /\/\*[^]*?\*\//g;
function ge(e) {
	let t = {};
	return e.replace(he, "").split(pe).forEach((e) => {
		if (e) {
			let n = e.split(me);
			n.length > 1 && (t[n[0].trim()] = n[1].trim());
		}
	}), t;
}
function O(e) {
	let t = "";
	if (g(e)) t = e;
	else if (d(e)) for (let n = 0; n < e.length; n++) {
		let r = O(e[n]);
		r && (t += r + " ");
	}
	else if (v(e)) for (let n in e) e[n] && (t += n + " ");
	return t.trim();
}
var _e = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", ve = /* @__PURE__ */ e(_e);
_e + "";
function ye(e) {
	return !!e || e === "";
}
function be(e, t) {
	if (e.length !== t.length) return !1;
	let n = !0;
	for (let r = 0; n && r < e.length; r++) n = xe(e[r], t[r]);
	return n;
}
function xe(e, t) {
	if (e === t) return !0;
	let n = m(e), r = m(t);
	if (n || r) return n && r ? e.getTime() === t.getTime() : !1;
	if (n = _(e), r = _(t), n || r) return e === t;
	if (n = d(e), r = d(t), n || r) return n && r ? be(e, t) : !1;
	if (n = v(e), r = v(t), n || r) {
		if (!n || !r || Object.keys(e).length !== Object.keys(t).length) return !1;
		for (let n in e) {
			let r = e.hasOwnProperty(n), i = t.hasOwnProperty(n);
			if (r && !i || !r && i || !xe(e[n], t[n])) return !1;
		}
	}
	return String(e) === String(t);
}
function Se(e, t) {
	return e.findIndex((e) => xe(e, t));
}
var Ce = (e) => !!(e && e.__v_isRef === !0), k = (e) => g(e) ? e : e == null ? "" : d(e) || v(e) && (e.toString === b || !h(e.toString)) ? Ce(e) ? k(e.value) : JSON.stringify(e, we, 2) : String(e), we = (e, t) => Ce(t) ? we(e, t.value) : f(t) ? { [`Map(${t.size})`]: [...t.entries()].reduce((e, [t, n], r) => (e[Te(t, r) + " =>"] = n, e), {}) } : p(t) ? { [`Set(${t.size})`]: [...t.values()].map((e) => Te(e)) } : _(t) ? Te(t) : v(t) && !d(t) && !C(t) ? String(t) : t, Te = (e, t = "") => _(e) ? `Symbol(${e.description ?? t})` : e, A, Ee = class {
	constructor(e = !1) {
		this.detached = e, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this._warnOnRun = !0, this.__v_skip = !0, !e && A && (A.active ? (this.parent = A, this.index = (A.scopes || (A.scopes = [])).push(this) - 1) : (this._active = !1, this._warnOnRun = !1));
	}
	get active() {
		return this._active;
	}
	pause() {
		if (this._active) {
			this._isPaused = !0;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].pause();
			}
			for (e = 0, t = this.effects.length; e < t; e++) this.effects[e].pause();
		}
	}
	resume() {
		if (this._active && this._isPaused) {
			this._isPaused = !1;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].resume();
			}
			let n = this.effects.slice();
			for (e = 0, t = n.length; e < t; e++) n[e].resume();
		}
	}
	run(e) {
		if (this._active) {
			let t = A;
			try {
				return A = this, e();
			} finally {
				A = t;
			}
		}
	}
	on() {
		++this._on === 1 && (this.prevScope = A, A = this);
	}
	off() {
		if (this._on > 0 && --this._on === 0) {
			if (A === this) A = this.prevScope;
			else {
				let e = A;
				for (; e;) {
					if (e.prevScope === this) {
						e.prevScope = this.prevScope;
						break;
					}
					e = e.prevScope;
				}
			}
			this.prevScope = void 0;
		}
	}
	stop(e) {
		if (this._active) {
			this._active = !1;
			let t, n;
			for (t = 0, n = this.effects.length; t < n; t++) this.effects[t].stop();
			for (this.effects.length = 0, t = 0, n = this.cleanups.length; t < n; t++) this.cleanups[t]();
			if (this.cleanups.length = 0, this.scopes) {
				let e = this.scopes.slice();
				for (t = 0, n = e.length; t < n; t++) e[t].stop(!0);
				this.scopes.length = 0;
			}
			if (!this.detached && this.parent && !e) {
				let e = this.parent.scopes.pop();
				e && e !== this && (this.parent.scopes[this.index] = e, e.index = this.index);
			}
			this.parent = void 0;
		}
	}
};
function De() {
	return A;
}
var j, Oe = /* @__PURE__ */ new WeakSet(), ke = class {
	constructor(e) {
		this.fn = e, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, A && (A.active ? A.effects.push(this) : this.flags &= -2);
	}
	pause() {
		this.flags |= 64;
	}
	resume() {
		this.flags & 64 && (this.flags &= -65, Oe.has(this) && (Oe.delete(this), this.trigger()));
	}
	notify() {
		this.flags & 2 && !(this.flags & 32) || this.flags & 8 || Ne(this);
	}
	run() {
		if (!(this.flags & 1)) return this.fn();
		this.flags |= 2, Ke(this), Ie(this);
		let e = j, t = He;
		j = this, He = !0;
		try {
			return this.fn();
		} finally {
			Le(this), j = e, He = t, this.flags &= -3;
		}
	}
	stop() {
		if (this.flags & 1) {
			for (let e = this.deps; e; e = e.nextDep) Be(e);
			this.deps = this.depsTail = void 0, Ke(this), this.onStop && this.onStop(), this.flags &= -2;
		}
	}
	trigger() {
		this.flags & 64 ? Oe.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
	}
	runIfDirty() {
		Re(this) && this.run();
	}
	get dirty() {
		return Re(this);
	}
}, Ae = 0, je, Me;
function Ne(e, t = !1) {
	if (e.flags |= 8, t) {
		e.next = Me, Me = e;
		return;
	}
	e.next = je, je = e;
}
function Pe() {
	Ae++;
}
function Fe() {
	if (--Ae > 0) return;
	if (Me) {
		let e = Me;
		for (Me = void 0; e;) {
			let t = e.next;
			e.next = void 0, e.flags &= -9, e = t;
		}
	}
	let e;
	for (; je;) {
		let t = je;
		for (je = void 0; t;) {
			let n = t.next;
			if (t.next = void 0, t.flags &= -9, t.flags & 1) try {
				t.trigger();
			} catch (t) {
				e ||= t;
			}
			t = n;
		}
	}
	if (e) throw e;
}
function Ie(e) {
	for (let t = e.deps; t; t = t.nextDep) t.version = -1, t.prevActiveLink = t.dep.activeLink, t.dep.activeLink = t;
}
function Le(e) {
	let t, n = e.depsTail, r = n;
	for (; r;) {
		let e = r.prevDep;
		r.version === -1 ? (r === n && (n = e), Be(r), Ve(r)) : t = r, r.dep.activeLink = r.prevActiveLink, r.prevActiveLink = void 0, r = e;
	}
	e.deps = t, e.depsTail = n;
}
function Re(e) {
	for (let t = e.deps; t; t = t.nextDep) if (t.dep.version !== t.version || t.dep.computed && (ze(t.dep.computed) || t.dep.version !== t.version)) return !0;
	return !!e._dirty;
}
function ze(e) {
	if (e.flags & 4 && !(e.flags & 16) || (e.flags &= -17, e.globalVersion === qe) || (e.globalVersion = qe, !e.isSSR && e.flags & 128 && (!e.deps && !e._dirty || !Re(e)))) return;
	e.flags |= 2;
	let t = e.dep, n = j, r = He;
	j = e, He = !0;
	try {
		Ie(e);
		let n = e.fn(e._value);
		(t.version === 0 || oe(n, e._value)) && (e.flags |= 128, e._value = n, t.version++);
	} catch (e) {
		throw t.version++, e;
	} finally {
		j = n, He = r, Le(e), e.flags &= -3;
	}
}
function Be(e, t = !1) {
	let { dep: n, prevSub: r, nextSub: i } = e;
	if (r && (r.nextSub = i, e.prevSub = void 0), i && (i.prevSub = r, e.nextSub = void 0), n.subs === e && (n.subs = r, !r && n.computed)) {
		n.computed.flags &= -5;
		for (let e = n.computed.deps; e; e = e.nextDep) Be(e, !0);
	}
	!t && !--n.sc && n.map && n.map.delete(n.key);
}
function Ve(e) {
	let { prevDep: t, nextDep: n } = e;
	t && (t.nextDep = n, e.prevDep = void 0), n && (n.prevDep = t, e.nextDep = void 0);
}
var He = !0, Ue = [];
function We() {
	Ue.push(He), He = !1;
}
function Ge() {
	let e = Ue.pop();
	He = e === void 0 || e;
}
function Ke(e) {
	let { cleanup: t } = e;
	if (e.cleanup = void 0, t) {
		let e = j;
		j = void 0;
		try {
			t();
		} finally {
			j = e;
		}
	}
}
var qe = 0, Je = class {
	constructor(e, t) {
		this.sub = e, this.dep = t, this.version = t.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
	}
}, Ye = class {
	constructor(e) {
		this.computed = e, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0;
	}
	track(e) {
		if (!j || !He || j === this.computed) return;
		let t = this.activeLink;
		if (t === void 0 || t.sub !== j) t = this.activeLink = new Je(j, this), j.deps ? (t.prevDep = j.depsTail, j.depsTail.nextDep = t, j.depsTail = t) : j.deps = j.depsTail = t, Xe(t);
		else if (t.version === -1 && (t.version = this.version, t.nextDep)) {
			let e = t.nextDep;
			e.prevDep = t.prevDep, t.prevDep && (t.prevDep.nextDep = e), t.prevDep = j.depsTail, t.nextDep = void 0, j.depsTail.nextDep = t, j.depsTail = t, j.deps === t && (j.deps = e);
		}
		return t;
	}
	trigger(e) {
		this.version++, qe++, this.notify(e);
	}
	notify(e) {
		Pe();
		try {
			for (let e = this.subs; e; e = e.prevSub) e.sub.notify() && e.sub.dep.notify();
		} finally {
			Fe();
		}
	}
};
function Xe(e) {
	if (e.dep.sc++, e.sub.flags & 4) {
		let t = e.dep.computed;
		if (t && !e.dep.subs) {
			t.flags |= 20;
			for (let e = t.deps; e; e = e.nextDep) Xe(e);
		}
		let n = e.dep.subs;
		n !== e && (e.prevSub = n, n && (n.nextSub = e)), e.dep.subs = e;
	}
}
var Ze = /* @__PURE__ */ new WeakMap(), Qe = /* @__PURE__ */ Symbol(""), $e = /* @__PURE__ */ Symbol(""), et = /* @__PURE__ */ Symbol("");
function M(e, t, n) {
	if (He && j) {
		let t = Ze.get(e);
		t || Ze.set(e, t = /* @__PURE__ */ new Map());
		let r = t.get(n);
		r || (t.set(n, r = new Ye()), r.map = t, r.key = n), r.track();
	}
}
function tt(e, t, n, r, i, a) {
	let o = Ze.get(e);
	if (!o) {
		qe++;
		return;
	}
	let s = (e) => {
		e && e.trigger();
	};
	if (Pe(), t === "clear") o.forEach(s);
	else {
		let i = d(e), a = i && w(n);
		if (i && n === "length") {
			let e = Number(r);
			o.forEach((t, n) => {
				(n === "length" || n === et || !_(n) && n >= e) && s(t);
			});
		} else switch ((n !== void 0 || o.has(void 0)) && s(o.get(n)), a && s(o.get(et)), t) {
			case "add":
				i ? a && s(o.get("length")) : (s(o.get(Qe)), f(e) && s(o.get($e)));
				break;
			case "delete":
				i || (s(o.get(Qe)), f(e) && s(o.get($e)));
				break;
			case "set": f(e) && s(o.get(Qe));
		}
	}
	Fe();
}
function nt(e) {
	let t = /* @__PURE__ */ N(e);
	return t === e ? t : (M(t, "iterate", et), /* @__PURE__ */ Vt(e) ? t : t.map(Wt));
}
function rt(e) {
	return M(e = /* @__PURE__ */ N(e), "iterate", et), e;
}
function it(e, t) {
	return /* @__PURE__ */ Bt(e) ? Gt(/* @__PURE__ */ zt(e) ? Wt(t) : t) : Wt(t);
}
var at = {
	__proto__: null,
	[Symbol.iterator]() {
		return ot(this, Symbol.iterator, (e) => it(this, e));
	},
	concat(...e) {
		return nt(this).concat(...e.map((e) => d(e) ? nt(e) : e));
	},
	entries() {
		return ot(this, "entries", (e) => (e[1] = it(this, e[1]), e));
	},
	every(e, t) {
		return ct(this, "every", e, t, void 0, arguments);
	},
	filter(e, t) {
		return ct(this, "filter", e, t, (e) => e.map((e) => it(this, e)), arguments);
	},
	find(e, t) {
		return ct(this, "find", e, t, (e) => it(this, e), arguments);
	},
	findIndex(e, t) {
		return ct(this, "findIndex", e, t, void 0, arguments);
	},
	findLast(e, t) {
		return ct(this, "findLast", e, t, (e) => it(this, e), arguments);
	},
	findLastIndex(e, t) {
		return ct(this, "findLastIndex", e, t, void 0, arguments);
	},
	forEach(e, t) {
		return ct(this, "forEach", e, t, void 0, arguments);
	},
	includes(...e) {
		return ut(this, "includes", e);
	},
	indexOf(...e) {
		return ut(this, "indexOf", e);
	},
	join(e) {
		return nt(this).join(e);
	},
	lastIndexOf(...e) {
		return ut(this, "lastIndexOf", e);
	},
	map(e, t) {
		return ct(this, "map", e, t, void 0, arguments);
	},
	pop() {
		return dt(this, "pop");
	},
	push(...e) {
		return dt(this, "push", e);
	},
	reduce(e, ...t) {
		return lt(this, "reduce", e, t);
	},
	reduceRight(e, ...t) {
		return lt(this, "reduceRight", e, t);
	},
	shift() {
		return dt(this, "shift");
	},
	some(e, t) {
		return ct(this, "some", e, t, void 0, arguments);
	},
	splice(...e) {
		return dt(this, "splice", e);
	},
	toReversed() {
		return nt(this).toReversed();
	},
	toSorted(e) {
		return nt(this).toSorted(e);
	},
	toSpliced(...e) {
		return nt(this).toSpliced(...e);
	},
	unshift(...e) {
		return dt(this, "unshift", e);
	},
	values() {
		return ot(this, "values", (e) => it(this, e));
	}
};
function ot(e, t, n) {
	let r = rt(e), i = r[t]();
	return r !== e && !/* @__PURE__ */ Vt(e) && (i._next = i.next, i.next = () => {
		let e = i._next();
		return e.done || (e.value = n(e.value)), e;
	}), i;
}
var st = Array.prototype;
function ct(e, t, n, r, i, a) {
	let o = rt(e), s = o !== e && !/* @__PURE__ */ Vt(e), c = o[t];
	if (c !== st[t]) {
		let t = c.apply(e, a);
		return s ? Wt(t) : t;
	}
	let l = n;
	o !== e && (s ? l = function(t, r) {
		return n.call(this, it(e, t), r, e);
	} : n.length > 2 && (l = function(t, r) {
		return n.call(this, t, r, e);
	}));
	let u = c.call(o, l, r);
	return s && i ? i(u) : u;
}
function lt(e, t, n, r) {
	let i = rt(e), a = i !== e && !/* @__PURE__ */ Vt(e), o = n, s = !1;
	i !== e && (a ? (s = r.length === 0, o = function(t, r, i) {
		return s && (s = !1, t = it(e, t)), n.call(this, t, it(e, r), i, e);
	}) : n.length > 3 && (o = function(t, r, i) {
		return n.call(this, t, r, i, e);
	}));
	let c = i[t](o, ...r);
	return s ? it(e, c) : c;
}
function ut(e, t, n) {
	let r = /* @__PURE__ */ N(e);
	M(r, "iterate", et);
	let i = r[t](...n);
	return (i === -1 || i === !1) && /* @__PURE__ */ Ht(n[0]) ? (n[0] = /* @__PURE__ */ N(n[0]), r[t](...n)) : i;
}
function dt(e, t, n = []) {
	We(), Pe();
	let r = (/* @__PURE__ */ N(e))[t].apply(e, n);
	return Fe(), Ge(), r;
}
var ft = /* @__PURE__ */ e("__proto__,__v_isRef,__isVue"), pt = new Set(/* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((e) => e !== "arguments" && e !== "caller").map((e) => Symbol[e]).filter(_));
function mt(e) {
	_(e) || (e = String(e));
	let t = /* @__PURE__ */ N(this);
	return M(t, "has", e), t.hasOwnProperty(e);
}
var ht = class {
	constructor(e = !1, t = !1) {
		this._isReadonly = e, this._isShallow = t;
	}
	get(e, t, n) {
		if (t === "__v_skip") return e.__v_skip;
		let r = this._isReadonly, i = this._isShallow;
		if (t === "__v_isReactive") return !r;
		if (t === "__v_isReadonly") return r;
		if (t === "__v_isShallow") return i;
		if (t === "__v_raw") return n === (r ? i ? Nt : Mt : i ? jt : At).get(e) || Object.getPrototypeOf(e) === Object.getPrototypeOf(n) ? e : void 0;
		let a = d(e);
		if (!r) {
			let e;
			if (a && (e = at[t])) return e;
			if (t === "hasOwnProperty") return mt;
		}
		let o = Reflect.get(e, t, /* @__PURE__ */ P(e) ? e : n);
		if ((_(t) ? pt.has(t) : ft(t)) || (r || M(e, "get", t), i)) return o;
		if (/* @__PURE__ */ P(o)) {
			let e = a && w(t) ? o : o.value;
			return r && v(e) ? /* @__PURE__ */ Lt(e) : e;
		}
		return v(o) ? r ? /* @__PURE__ */ Lt(o) : /* @__PURE__ */ Ft(o) : o;
	}
}, gt = class extends ht {
	constructor(e = !1) {
		super(!1, e);
	}
	set(e, t, n, r) {
		let i = e[t], a = d(e) && w(t);
		if (!this._isShallow) {
			let e = /* @__PURE__ */ Bt(i);
			if (!/* @__PURE__ */ Vt(n) && !/* @__PURE__ */ Bt(n) && (i = /* @__PURE__ */ N(i), n = /* @__PURE__ */ N(n)), !a && /* @__PURE__ */ P(i) && !/* @__PURE__ */ P(n)) return e || (i.value = n), !0;
		}
		let o = a ? Number(t) < e.length : u(e, t), s = Reflect.set(e, t, n, /* @__PURE__ */ P(e) ? e : r);
		return e === /* @__PURE__ */ N(r) && s && (o ? oe(n, i) && tt(e, "set", t, n, i) : tt(e, "add", t, n)), s;
	}
	deleteProperty(e, t) {
		let n = u(e, t), r = e[t], i = Reflect.deleteProperty(e, t);
		return i && n && tt(e, "delete", t, void 0, r), i;
	}
	has(e, t) {
		let n = Reflect.has(e, t);
		return (!_(t) || !pt.has(t)) && M(e, "has", t), n;
	}
	ownKeys(e) {
		return M(e, "iterate", d(e) ? "length" : Qe), Reflect.ownKeys(e);
	}
}, _t = class extends ht {
	constructor(e = !1) {
		super(!0, e);
	}
	set(e, t) {
		return !0;
	}
	deleteProperty(e, t) {
		return !0;
	}
}, vt = /* @__PURE__ */ new gt(), yt = /* @__PURE__ */ new _t(), bt = /* @__PURE__ */ new gt(!0), xt = (e) => e, St = (e) => Reflect.getPrototypeOf(e);
function Ct(e, t, n) {
	return function(...r) {
		let i = this.__v_raw, a = /* @__PURE__ */ N(i), o = f(a), c = e === "entries" || e === Symbol.iterator && o, l = e === "keys" && o, u = i[e](...r), d = n ? xt : t ? Gt : Wt;
		return !t && M(a, "iterate", l ? $e : Qe), s(Object.create(u), { next() {
			let { value: e, done: t } = u.next();
			return t ? {
				value: e,
				done: t
			} : {
				value: c ? [d(e[0]), d(e[1])] : d(e),
				done: t
			};
		} });
	};
}
function wt(e) {
	return function(...t) {
		return e === "delete" ? !1 : e === "clear" ? void 0 : this;
	};
}
function Tt(e, t) {
	let n = {
		get(n) {
			let r = this.__v_raw, i = /* @__PURE__ */ N(r), a = /* @__PURE__ */ N(n);
			e || (oe(n, a) && M(i, "get", n), M(i, "get", a));
			let { has: o } = St(i), s = t ? xt : e ? Gt : Wt;
			if (o.call(i, n)) return s(r.get(n));
			if (o.call(i, a)) return s(r.get(a));
			r !== i && r.get(n);
		},
		get size() {
			let t = this.__v_raw;
			return !e && M(/* @__PURE__ */ N(t), "iterate", Qe), t.size;
		},
		has(t) {
			let n = this.__v_raw, r = /* @__PURE__ */ N(n), i = /* @__PURE__ */ N(t);
			return e || (oe(t, i) && M(r, "has", t), M(r, "has", i)), t === i ? n.has(t) : n.has(t) || n.has(i);
		},
		forEach(n, r) {
			let i = this, a = i.__v_raw, o = /* @__PURE__ */ N(a), s = t ? xt : e ? Gt : Wt;
			return !e && M(o, "iterate", Qe), a.forEach((e, t) => n.call(r, s(e), s(t), i));
		}
	};
	return s(n, e ? {
		add: wt("add"),
		set: wt("set"),
		delete: wt("delete"),
		clear: wt("clear")
	} : {
		add(e) {
			let n = /* @__PURE__ */ N(this), r = St(n), i = /* @__PURE__ */ N(e), a = !t && !/* @__PURE__ */ Vt(e) && !/* @__PURE__ */ Bt(e) ? i : e;
			return r.has.call(n, a) || oe(e, a) && r.has.call(n, e) || oe(i, a) && r.has.call(n, i) || (n.add(a), tt(n, "add", a, a)), this;
		},
		set(e, n) {
			!t && !/* @__PURE__ */ Vt(n) && !/* @__PURE__ */ Bt(n) && (n = /* @__PURE__ */ N(n));
			let r = /* @__PURE__ */ N(this), { has: i, get: a } = St(r), o = i.call(r, e);
			o ||= (e = /* @__PURE__ */ N(e), i.call(r, e));
			let s = a.call(r, e);
			return r.set(e, n), o ? oe(n, s) && tt(r, "set", e, n, s) : tt(r, "add", e, n), this;
		},
		delete(e) {
			let t = /* @__PURE__ */ N(this), { has: n, get: r } = St(t), i = n.call(t, e);
			i ||= (e = /* @__PURE__ */ N(e), n.call(t, e));
			let a = r ? r.call(t, e) : void 0, o = t.delete(e);
			return i && tt(t, "delete", e, void 0, a), o;
		},
		clear() {
			let e = /* @__PURE__ */ N(this), t = e.size !== 0, n = e.clear();
			return t && tt(e, "clear", void 0, void 0, void 0), n;
		}
	}), [
		"keys",
		"values",
		"entries",
		Symbol.iterator
	].forEach((r) => {
		n[r] = Ct(r, e, t);
	}), n;
}
function Et(e, t) {
	let n = Tt(e, t);
	return (t, r, i) => r === "__v_isReactive" ? !e : r === "__v_isReadonly" ? e : r === "__v_raw" ? t : Reflect.get(u(n, r) && r in t ? n : t, r, i);
}
var Dt = { get: /* @__PURE__ */ Et(!1, !1) }, Ot = { get: /* @__PURE__ */ Et(!1, !0) }, kt = { get: /* @__PURE__ */ Et(!0, !1) }, At = /* @__PURE__ */ new WeakMap(), jt = /* @__PURE__ */ new WeakMap(), Mt = /* @__PURE__ */ new WeakMap(), Nt = /* @__PURE__ */ new WeakMap();
function Pt(e) {
	switch (e) {
		case "Object":
		case "Array": return 1;
		case "Map":
		case "Set":
		case "WeakMap":
		case "WeakSet": return 2;
		default: return 0;
	}
}
// @__NO_SIDE_EFFECTS__
function Ft(e) {
	return /* @__PURE__ */ Bt(e) ? e : Rt(e, !1, vt, Dt, At);
}
// @__NO_SIDE_EFFECTS__
function It(e) {
	return Rt(e, !1, bt, Ot, jt);
}
// @__NO_SIDE_EFFECTS__
function Lt(e) {
	return Rt(e, !0, yt, kt, Mt);
}
function Rt(e, t, n, r, i) {
	if (!v(e) || e.__v_raw && !(t && e.__v_isReactive) || e.__v_skip || !Object.isExtensible(e)) return e;
	let a = i.get(e);
	if (a) return a;
	let o = Pt(S(e));
	if (o === 0) return e;
	let s = new Proxy(e, o === 2 ? r : n);
	return i.set(e, s), s;
}
// @__NO_SIDE_EFFECTS__
function zt(e) {
	return /* @__PURE__ */ Bt(e) ? /* @__PURE__ */ zt(e.__v_raw) : !!(e && e.__v_isReactive);
}
// @__NO_SIDE_EFFECTS__
function Bt(e) {
	return !!(e && e.__v_isReadonly);
}
// @__NO_SIDE_EFFECTS__
function Vt(e) {
	return !!(e && e.__v_isShallow);
}
// @__NO_SIDE_EFFECTS__
function Ht(e) {
	return e ? !!e.__v_raw : !1;
}
// @__NO_SIDE_EFFECTS__
function N(e) {
	let t = e && e.__v_raw;
	return t ? /* @__PURE__ */ N(t) : e;
}
function Ut(e) {
	return !u(e, "__v_skip") && Object.isExtensible(e) && D(e, "__v_skip", !0), e;
}
var Wt = (e) => v(e) ? /* @__PURE__ */ Ft(e) : e, Gt = (e) => v(e) ? /* @__PURE__ */ Lt(e) : e;
// @__NO_SIDE_EFFECTS__
function P(e) {
	return e ? e.__v_isRef === !0 : !1;
}
// @__NO_SIDE_EFFECTS__
function F(e) {
	return Kt(e, !1);
}
function Kt(e, t) {
	return /* @__PURE__ */ P(e) ? e : new qt(e, t);
}
var qt = class {
	constructor(e, t) {
		this.dep = new Ye(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = t ? e : /* @__PURE__ */ N(e), this._value = t ? e : Wt(e), this.__v_isShallow = t;
	}
	get value() {
		return this.dep.track(), this._value;
	}
	set value(e) {
		let t = this._rawValue, n = this.__v_isShallow || /* @__PURE__ */ Vt(e) || /* @__PURE__ */ Bt(e);
		e = n ? e : /* @__PURE__ */ N(e), oe(e, t) && (this._rawValue = e, this._value = n ? e : Wt(e), this.dep.trigger());
	}
};
function I(e) {
	return /* @__PURE__ */ P(e) ? e.value : e;
}
var Jt = {
	get: (e, t, n) => t === "__v_raw" ? e : I(Reflect.get(e, t, n)),
	set: (e, t, n, r) => {
		let i = e[t];
		return /* @__PURE__ */ P(i) && !/* @__PURE__ */ P(n) ? (i.value = n, !0) : Reflect.set(e, t, n, r);
	}
};
function Yt(e) {
	return /* @__PURE__ */ zt(e) ? e : new Proxy(e, Jt);
}
var Xt = class {
	constructor(e, t, n) {
		this.fn = e, this.setter = t, this._value = void 0, this.dep = new Ye(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = qe - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !t, this.isSSR = n;
	}
	notify() {
		if (this.flags |= 16, !(this.flags & 8) && j !== this) return Ne(this, !0), !0;
	}
	get value() {
		let e = this.dep.track();
		return ze(this), e && (e.version = this.dep.version), this._value;
	}
	set value(e) {
		this.setter && this.setter(e);
	}
};
// @__NO_SIDE_EFFECTS__
function Zt(e, t, n = !1) {
	let r, i;
	return h(e) ? r = e : (r = e.get, i = e.set), new Xt(r, i, n);
}
var Qt = {}, $t = /* @__PURE__ */ new WeakMap(), en = void 0;
function tn(e, t = !1, n = en) {
	if (n) {
		let t = $t.get(n);
		t || $t.set(n, t = []), t.push(e);
	}
}
function nn(e, n, i = t) {
	let { immediate: a, deep: o, once: s, scheduler: l, augmentJob: u, call: f } = i, p = (e) => o ? e : /* @__PURE__ */ Vt(e) || o === !1 || o === 0 ? rn(e, 1) : rn(e), m, g, _, v, y = !1, b = !1;
	if (/* @__PURE__ */ P(e) ? (g = () => e.value, y = /* @__PURE__ */ Vt(e)) : /* @__PURE__ */ zt(e) ? (g = () => p(e), y = !0) : d(e) ? (b = !0, y = e.some((e) => /* @__PURE__ */ zt(e) || /* @__PURE__ */ Vt(e)), g = () => e.map((e) => {
		if (/* @__PURE__ */ P(e)) return e.value;
		if (/* @__PURE__ */ zt(e)) return p(e);
		if (h(e)) return f ? f(e, 2) : e();
	})) : g = h(e) ? n ? f ? () => f(e, 2) : e : () => {
		if (_) {
			We();
			try {
				_();
			} finally {
				Ge();
			}
		}
		let t = en;
		en = m;
		try {
			return f ? f(e, 3, [v]) : e(v);
		} finally {
			en = t;
		}
	} : r, n && o) {
		let e = g, t = o === !0 ? Infinity : o;
		g = () => rn(e(), t);
	}
	let x = De(), S = () => {
		m.stop(), x && x.active && c(x.effects, m);
	};
	if (s && n) {
		let e = n;
		n = (...t) => {
			let n = e(...t);
			return S(), n;
		};
	}
	let C = b ? Array(e.length).fill(Qt) : Qt, w = (e) => {
		if (!(!(m.flags & 1) || !m.dirty && !e)) if (n) {
			let t = m.run();
			if (e || o || y || (b ? t.some((e, t) => oe(e, C[t])) : oe(t, C))) {
				_ && _();
				let e = en;
				en = m;
				try {
					let e = [
						t,
						C === Qt ? void 0 : b && C[0] === Qt ? [] : C,
						v
					];
					C = t, f ? f(n, 3, e) : n(...e);
				} finally {
					en = e;
				}
			}
		} else m.run();
	};
	return u && u(w), m = new ke(g), m.scheduler = l ? () => l(w, !1) : w, v = (e) => tn(e, !1, m), _ = m.onStop = () => {
		let e = $t.get(m);
		if (e) {
			if (f) f(e, 4);
			else for (let t of e) t();
			$t.delete(m);
		}
	}, n ? a ? w(!0) : C = m.run() : l ? l(w.bind(null, !0), !0) : m.run(), S.pause = m.pause.bind(m), S.resume = m.resume.bind(m), S.stop = S, S;
}
function rn(e, t = Infinity, n) {
	if (t <= 0 || !v(e) || e.__v_skip || (n ||= /* @__PURE__ */ new Map(), (n.get(e) || 0) >= t)) return e;
	if (n.set(e, t), t--, /* @__PURE__ */ P(e)) rn(e.value, t, n);
	else if (d(e)) for (let r = 0; r < e.length; r++) rn(e[r], t, n);
	else if (p(e) || f(e)) e.forEach((e) => {
		rn(e, t, n);
	});
	else if (C(e)) {
		for (let r in e) rn(e[r], t, n);
		for (let r of Object.getOwnPropertySymbols(e)) Object.prototype.propertyIsEnumerable.call(e, r) && rn(e[r], t, n);
	}
	return e;
}
//#endregion
//#region node_modules/@vue/runtime-core/dist/runtime-core.esm-bundler.js
function an(e, t, n, r) {
	try {
		return r ? e(...r) : e();
	} catch (e) {
		sn(e, t, n);
	}
}
function on(e, t, n, r) {
	if (h(e)) {
		let i = an(e, t, n, r);
		return i && y(i) && i.catch((e) => {
			sn(e, t, n);
		}), i;
	}
	if (d(e)) {
		let i = [];
		for (let a = 0; a < e.length; a++) i.push(on(e[a], t, n, r));
		return i;
	}
}
function sn(e, n, r, i = !0) {
	let a = n ? n.vnode : null, { errorHandler: o, throwUnhandledErrorInProduction: s } = n && n.appContext.config || t;
	if (n) {
		let t = n.parent, i = n.proxy, a = `https://vuejs.org/error-reference/#runtime-${r}`;
		for (; t;) {
			let n = t.ec;
			if (n) {
				for (let t = 0; t < n.length; t++) if (n[t](e, i, a) === !1) return;
			}
			t = t.parent;
		}
		if (o) {
			We(), an(o, null, 10, [
				e,
				i,
				a
			]), Ge();
			return;
		}
	}
	cn(e, r, a, i, s);
}
function cn(e, t, n, r = !0, i = !1) {
	if (i) throw e;
	console.error(e);
}
var L = [], ln = -1, un = [], dn = null, fn = 0, pn = /* @__PURE__ */ Promise.resolve(), mn = null;
function hn(e) {
	let t = mn || pn;
	return e ? t.then(this ? e.bind(this) : e) : t;
}
function gn(e) {
	let t = ln + 1, n = L.length;
	for (; t < n;) {
		let r = t + n >>> 1, i = L[r], a = Sn(i);
		a < e || a === e && i.flags & 2 ? t = r + 1 : n = r;
	}
	return t;
}
function _n(e) {
	if (!(e.flags & 1)) {
		let t = Sn(e), n = L[L.length - 1];
		!n || !(e.flags & 2) && t >= Sn(n) ? L.push(e) : L.splice(gn(t), 0, e), e.flags |= 1, vn();
	}
}
function vn() {
	mn ||= pn.then(Cn);
}
function yn(e) {
	d(e) ? un.push(...e) : dn && e.id === -1 ? dn.splice(fn + 1, 0, e) : e.flags & 1 || (un.push(e), e.flags |= 1), vn();
}
function bn(e, t, n = ln + 1) {
	for (; n < L.length; n++) {
		let t = L[n];
		if (t && t.flags & 2) {
			if (e && t.id !== e.uid) continue;
			L.splice(n, 1), n--, t.flags & 4 && (t.flags &= -2), t(), t.flags & 4 || (t.flags &= -2);
		}
	}
}
function xn(e) {
	if (un.length) {
		let e = [...new Set(un)].sort((e, t) => Sn(e) - Sn(t));
		if (un.length = 0, dn) {
			dn.push(...e);
			return;
		}
		for (dn = e, fn = 0; fn < dn.length; fn++) {
			let e = dn[fn];
			e.flags & 4 && (e.flags &= -2), e.flags & 8 || e(), e.flags &= -2;
		}
		dn = null, fn = 0;
	}
}
var Sn = (e) => e.id == null ? e.flags & 2 ? -1 : Infinity : e.id;
function Cn(e) {
	try {
		for (ln = 0; ln < L.length; ln++) {
			let e = L[ln];
			e && !(e.flags & 8) && (e.flags & 4 && (e.flags &= -2), an(e, e.i, e.i ? 15 : 14), e.flags & 4 || (e.flags &= -2));
		}
	} finally {
		for (; ln < L.length; ln++) {
			let e = L[ln];
			e && (e.flags &= -2);
		}
		ln = -1, L.length = 0, xn(e), mn = null, (L.length || un.length) && Cn(e);
	}
}
var wn = null, Tn = null;
function En(e) {
	let t = wn;
	return wn = e, Tn = e && e.type.__scopeId || null, t;
}
function Dn(e, t = wn, n) {
	if (!t || e._n) return e;
	let r = (...n) => {
		r._d && ra(-1);
		let i = En(t), a = $i.length, o;
		try {
			o = e(...n);
		} finally {
			for (let e = $i.length; e > a; e--) ta();
			En(i), r._d && ra(1);
		}
		return o;
	};
	return r._n = !0, r._c = !0, r._d = !0, r;
}
function R(e, n) {
	if (wn === null) return e;
	let r = Ra(wn), i = e.dirs ||= [];
	for (let e = 0; e < n.length; e++) {
		let [a, o, s, c = t] = n[e];
		a && (h(a) && (a = {
			mounted: a,
			updated: a
		}), a.deep && rn(o), i.push({
			dir: a,
			instance: r,
			value: o,
			oldValue: void 0,
			arg: s,
			modifiers: c
		}));
	}
	return e;
}
function On(e, t, n, r) {
	let i = e.dirs, a = t && t.dirs;
	for (let o = 0; o < i.length; o++) {
		let s = i[o];
		a && (s.oldValue = a[o].value);
		let c = s.dir[r];
		c && (We(), on(c, n, 8, [
			e.el,
			s,
			e,
			t
		]), Ge());
	}
}
function kn(e, t) {
	if (Sa) {
		let n = Sa.provides, r = Sa.parent && Sa.parent.provides;
		r === n && (n = Sa.provides = Object.create(r)), n[e] = t;
	}
}
function An(e, t, n = !1) {
	let r = Ca();
	if (r || oi) {
		let i = oi ? oi._context.provides : r ? r.parent == null || r.ce ? r.vnode.appContext && r.vnode.appContext.provides : r.parent.provides : void 0;
		if (i && e in i) return i[e];
		if (arguments.length > 1) return n && h(t) ? t.call(r && r.proxy) : t;
	}
}
var jn = /* @__PURE__ */ Symbol.for("v-scx"), Mn = () => An(jn);
function Nn(e, t, n) {
	return Pn(e, t, n);
}
function Pn(e, n, i = t) {
	let { immediate: a, deep: o, flush: c, once: l } = i, u = s({}, i), d = n && a || !n && c !== "post", f;
	if (ka) {
		if (c === "sync") {
			let e = Mn();
			f = e.__watcherHandles ||= [];
		} else if (!d) {
			let e = () => {};
			return e.stop = r, e.resume = r, e.pause = r, e;
		}
	}
	let p = Sa;
	u.call = (e, t, n) => on(e, p, t, n);
	let m = !1;
	c === "post" ? u.scheduler = (e) => {
		Li(e, p && p.suspense);
	} : c !== "sync" && (m = !0, u.scheduler = (e, t) => {
		t ? e() : _n(e);
	}), u.augmentJob = (e) => {
		n && (e.flags |= 4), m && (e.flags |= 2, p && (e.id = p.uid, e.i = p));
	};
	let h = nn(e, n, u);
	return ka && (f ? f.push(h) : d && h()), h;
}
function Fn(e, t, n) {
	let r = this.proxy, i = g(e) ? e.includes(".") ? In(r, e) : () => r[e] : e.bind(r, r), a;
	h(t) ? a = t : (a = t.handler, n = t);
	let o = Ea(this), s = Pn(i, a.bind(r), n);
	return o(), s;
}
function In(e, t) {
	let n = t.split(".");
	return () => {
		let t = e;
		for (let e = 0; e < n.length && t; e++) t = t[n[e]];
		return t;
	};
}
var Ln = /* @__PURE__ */ new WeakMap(), Rn = /* @__PURE__ */ Symbol("_vte"), zn = (e) => e.__isTeleport, Bn = (e) => e && (e.disabled || e.disabled === ""), Vn = (e) => e && (e.defer || e.defer === ""), Hn = (e) => typeof SVGElement < "u" && e instanceof SVGElement, Un = (e) => typeof MathMLElement == "function" && e instanceof MathMLElement, Wn = (e, t) => {
	let n = e && e.to;
	return g(n) ? t ? t(n) : null : n;
}, Gn = {
	name: "Teleport",
	__isTeleport: !0,
	process(e, t, n, r, i, a, o, s, c, l) {
		let { mc: u, pc: d, pbc: f, o: { insert: p, querySelector: m, createText: h, createComment: g, parentNode: _ } } = l, v = Bn(t.props), { dynamicChildren: y } = t, b = (e, t, n) => {
			e.shapeFlag & 16 && u(e.children, t, n, i, a, o, s, c);
		}, x = (e = t) => {
			let n = Bn(e.props), r = e.target = Wn(e.props, m), a = Xn(r, e, h, p);
			r && (o !== "svg" && Hn(r) ? o = "svg" : o !== "mathml" && Un(r) && (o = "mathml"), i && i.isCE && (i.ce._teleportTargets || (i.ce._teleportTargets = /* @__PURE__ */ new Set())).add(r), n || (b(e, r, a), Yn(e, !1)));
		}, S = (e) => {
			let t = () => {
				if (Ln.get(e) === t) {
					if (Ln.delete(e), Bn(e.props)) {
						let t = _(e.el) || n;
						b(e, t, e.anchor), Yn(e, !0);
					}
					x(e);
				}
			};
			Ln.set(e, t), Li(t, a);
		};
		if (e == null) {
			let e = t.el = h(""), i = t.anchor = h("");
			if (p(e, n, r), p(i, n, r), Vn(t.props) || a && a.pendingBranch) {
				S(t);
				return;
			}
			v && (b(t, n, i), Yn(t, !0)), x();
		} else {
			t.el = e.el;
			let r = t.anchor = e.anchor, u = Ln.get(e);
			if (u) {
				u.flags |= 8, Ln.delete(e), S(t);
				return;
			}
			t.targetStart = e.targetStart;
			let p = t.target = e.target, h = t.targetAnchor = e.targetAnchor, g = Bn(e.props), _ = g ? n : p, b = g ? r : h;
			if (o === "svg" || Hn(p) ? o = "svg" : (o === "mathml" || Un(p)) && (o = "mathml"), y ? (f(e.dynamicChildren, y, _, i, a, o, s), Ui(e, t, !0)) : c || d(e, t, _, b, i, a, o, s, !1), v) g ? t.props && e.props && t.props.to !== e.props.to && (t.props.to = e.props.to) : Kn(t, n, r, l, 1);
			else if ((t.props && t.props.to) !== (e.props && e.props.to)) {
				let e = Wn(t.props, m);
				e && (t.target = e, Kn(t, e, null, l, 0));
			} else g && Kn(t, p, h, l, 1);
			Yn(t, v);
		}
	},
	remove(e, t, n, { um: r, o: { remove: i } }, a) {
		let { shapeFlag: o, children: s, anchor: c, targetStart: l, targetAnchor: u, target: d, props: f } = e, p = Bn(f), m = a || !p, h = Ln.get(e);
		if (h && (h.flags |= 8, Ln.delete(e)), d && (i(l), i(u)), a && i(c), !h && (p || d) && o & 16) for (let e = 0; e < s.length; e++) {
			let i = s[e];
			r(i, t, n, m, !!i.dynamicChildren);
		}
	},
	move: Kn,
	hydrate: qn
};
function Kn(e, t, n, { o: { insert: r }, m: i }, a = 2) {
	a === 0 && r(e.targetAnchor, t, n);
	let { el: o, anchor: s, shapeFlag: c, children: l, props: u } = e, d = a === 2;
	if (d && r(o, t, n), !Ln.has(e) && (!d || Bn(u)) && c & 16) for (let e = 0; e < l.length; e++) i(l[e], t, n, 2);
	d && r(s, t, n);
}
function qn(e, t, n, r, i, a, { o: { nextSibling: o, parentNode: s, querySelector: c, insert: l, createText: u } }, d) {
	function f(e, n) {
		let r = n;
		for (; r;) {
			if (r && r.nodeType === 8) {
				if (r.data === "teleport start anchor") t.targetStart = r;
				else if (r.data === "teleport anchor") {
					t.targetAnchor = r, e._lpa = t.targetAnchor && o(t.targetAnchor);
					break;
				}
			}
			r = o(r);
		}
	}
	function p(e, t) {
		t.anchor = d(o(e), t, s(e), n, r, i, a);
	}
	let m = t.target = Wn(t.props, c), h = Bn(t.props);
	if (m) {
		let c = m._lpa || m.firstChild;
		t.shapeFlag & 16 && (h ? (p(e, t), f(m, c), t.targetAnchor || Xn(m, t, u, l, s(e) === m ? e : null)) : (t.anchor = o(e), f(m, c), t.targetAnchor || Xn(m, t, u, l), d(c && o(c), t, m, n, r, i, a))), Yn(t, h);
	} else h && t.shapeFlag & 16 && (p(e, t), t.targetStart = e, t.targetAnchor = o(e));
	return t.anchor && o(t.anchor);
}
var Jn = Gn;
function Yn(e, t) {
	let n = e.ctx;
	if (n && n.ut) {
		let r, i;
		for (t ? (r = e.el, i = e.anchor) : (r = e.targetStart, i = e.targetAnchor); r && r !== i;) r.nodeType === 1 && r.setAttribute("data-v-owner", n.uid), r = r.nextSibling;
		n.ut();
	}
}
function Xn(e, t, n, r, i = null) {
	let a = t.targetStart = n(""), o = t.targetAnchor = n("");
	return a[Rn] = o, e && (r(a, e, i), r(o, e, i)), o;
}
var Zn = /* @__PURE__ */ Symbol("_leaveCb"), Qn = /* @__PURE__ */ Symbol("_enterCb");
function $n() {
	let e = {
		isMounted: !1,
		isLeaving: !1,
		isUnmounting: !1,
		leavingVNodes: /* @__PURE__ */ new Map()
	};
	return Dr(() => {
		e.isMounted = !0;
	}), Ar(() => {
		e.isUnmounting = !0;
	}), e;
}
var er = [Function, Array], tr = {
	mode: String,
	appear: Boolean,
	persisted: Boolean,
	onBeforeEnter: er,
	onEnter: er,
	onAfterEnter: er,
	onEnterCancelled: er,
	onBeforeLeave: er,
	onLeave: er,
	onAfterLeave: er,
	onLeaveCancelled: er,
	onBeforeAppear: er,
	onAppear: er,
	onAfterAppear: er,
	onAppearCancelled: er
}, nr = (e) => {
	let t = e.subTree;
	return t.component ? nr(t.component) : t;
}, rr = {
	name: "BaseTransition",
	props: tr,
	setup(e, { slots: t }) {
		let n = Ca(), r = $n();
		return () => {
			let i = t.default && dr(t.default(), !0), a = i && i.length ? ir(i) : n.subTree ? K() : void 0;
			if (!a) return;
			let o = /* @__PURE__ */ N(e), { mode: s } = o;
			if (r.isLeaving) return cr(a);
			let c = lr(a);
			if (!c) return cr(a);
			let l = sr(c, o, r, n, (e) => l = e);
			c.type !== Zi && ur(c, l);
			let u = n.subTree && lr(n.subTree);
			if (u && u.type !== Zi && !sa(u, c) && nr(n).type !== Zi) {
				let e = sr(u, o, r, n);
				if (ur(u, e), s === "out-in" && c.type !== Zi) return r.isLeaving = !0, e.afterLeave = () => {
					r.isLeaving = !1, n.job.flags & 8 || n.update(), delete e.afterLeave, u = void 0;
				}, cr(a);
				s === "in-out" && c.type !== Zi ? e.delayLeave = (e, t, n) => {
					let i = or(r, u);
					i[String(u.key)] = u, e[Zn] = () => {
						t(), e[Zn] = void 0, delete l.delayedLeave, u = void 0;
					}, l.delayedLeave = () => {
						n(), delete l.delayedLeave, u = void 0;
					};
				} : u = void 0;
			} else u &&= void 0;
			return a;
		};
	}
};
function ir(e) {
	let t = e[0];
	if (e.length > 1) {
		for (let n of e) if (n.type !== Zi) {
			t = n;
			break;
		}
	}
	return t;
}
var ar = rr;
function or(e, t) {
	let { leavingVNodes: n } = e, r = n.get(t.type);
	return r || (r = /* @__PURE__ */ Object.create(null), n.set(t.type, r)), r;
}
function sr(e, t, n, r, i) {
	let { appear: a, mode: o, persisted: s = !1, onBeforeEnter: c, onEnter: l, onAfterEnter: u, onEnterCancelled: f, onBeforeLeave: p, onLeave: m, onAfterLeave: h, onLeaveCancelled: g, onBeforeAppear: _, onAppear: v, onAfterAppear: y, onAppearCancelled: b } = t, x = String(e.key), S = or(n, e), C = (e, t) => {
		e && on(e, r, 9, t);
	}, w = (e, t) => {
		let n = t[1];
		C(e, t), d(e) ? e.every((e) => e.length <= 1) && n() : e.length <= 1 && n();
	}, ee = {
		mode: o,
		persisted: s,
		beforeEnter(t) {
			let r = c;
			if (!n.isMounted) if (a) r = _ || c;
			else return;
			t[Zn] && t[Zn](!0);
			let i = S[x];
			i && sa(e, i) && i.el[Zn] && i.el[Zn](), C(r, [t]);
		},
		enter(t) {
			if (S[x] === e) return;
			let r = l, i = u, o = f;
			if (!n.isMounted) if (a) r = v || l, i = y || u, o = b || f;
			else return;
			let s = !1;
			t[Qn] = (e) => {
				s || (s = !0, C(e ? o : i, [t]), ee.delayedLeave && ee.delayedLeave(), t[Qn] = void 0);
			};
			let c = t[Qn].bind(null, !1);
			r ? w(r, [t, c]) : c();
		},
		leave(t, r) {
			let i = String(e.key);
			if (t[Qn] && t[Qn](!0), n.isUnmounting) return r();
			C(p, [t]);
			let a = !1;
			t[Zn] = (n) => {
				a || (a = !0, r(), C(n ? g : h, [t]), t[Zn] = void 0, S[i] === e && delete S[i]);
			};
			let o = t[Zn].bind(null, !1);
			S[i] = e, m ? w(m, [t, o]) : o();
		},
		clone(e) {
			let a = sr(e, t, n, r, i);
			return i && i(a), a;
		}
	};
	return ee;
}
function cr(e) {
	if (yr(e)) return e = fa(e), e.children = null, e;
}
function lr(e) {
	if (!yr(e)) return zn(e.type) && e.children ? ir(e.children) : e;
	if (e.component) return e.component.subTree;
	let { shapeFlag: t, children: n } = e;
	if (n) {
		if (t & 16) return n[0];
		if (t & 32 && h(n.default)) return n.default();
	}
}
function ur(e, t) {
	e.shapeFlag & 6 && e.component ? (e.transition = t, ur(e.component.subTree, t)) : e.shapeFlag & 128 ? (e.ssContent.transition = t.clone(e.ssContent), e.ssFallback.transition = t.clone(e.ssFallback)) : e.transition = t;
}
function dr(e, t = !1, n) {
	let r = [], i = 0;
	for (let a = 0; a < e.length; a++) {
		let o = e[a], s = n == null ? o.key : String(n) + String(o.key == null ? a : o.key);
		o.type === B ? (o.patchFlag & 128 && i++, r = r.concat(dr(o.children, t, s))) : (t || o.type !== Zi) && r.push(s == null ? o : fa(o, { key: s }));
	}
	if (i > 1) for (let e = 0; e < r.length; e++) r[e].patchFlag = -2;
	return r;
}
// @__NO_SIDE_EFFECTS__
function fr(e, t) {
	return h(e) ? /* @__PURE__ */ s({ name: e.name }, t, { setup: e }) : e;
}
function pr(e) {
	e.ids = [
		e.ids[0] + e.ids[2]++ + "-",
		0,
		0
	];
}
function mr(e, t) {
	let n;
	return !!((n = Object.getOwnPropertyDescriptor(e, t)) && !n.configurable);
}
var hr = /* @__PURE__ */ new WeakMap();
function gr(e, n, r, a, o = !1) {
	if (d(e)) {
		e.forEach((e, t) => gr(e, n && (d(n) ? n[t] : n), r, a, o));
		return;
	}
	if (vr(a) && !o) {
		a.shapeFlag & 512 && a.type.__asyncResolved && a.component.subTree.component && gr(e, n, r, a.component.subTree);
		return;
	}
	let s = a.shapeFlag & 4 ? Ra(a.component) : a.el, l = o ? null : s, { i: f, r: p } = e, m = n && n.r, _ = f.refs === t ? f.refs = {} : f.refs, v = f.setupState, y = /* @__PURE__ */ N(v), b = v === t ? i : (e) => !mr(_, e) && u(y, e), x = (e, t) => !(t && mr(_, t));
	if (m != null && m !== p) {
		if (_r(n), g(m)) _[m] = null, b(m) && (v[m] = null);
		else if (/* @__PURE__ */ P(m)) {
			let e = n;
			x(m, e.k) && (m.value = null), e.k && (_[e.k] = null);
		}
	}
	if (h(p)) an(p, f, 12, [l, _]);
	else {
		let t = g(p), n = /* @__PURE__ */ P(p);
		if (t || n) {
			let i = () => {
				if (e.f) {
					let n = t ? b(p) ? v[p] : _[p] : x(p) || !e.k ? p.value : _[e.k];
					if (o) d(n) && c(n, s);
					else if (d(n)) n.includes(s) || n.push(s);
					else if (t) _[p] = [s], b(p) && (v[p] = _[p]);
					else {
						let t = [s];
						x(p, e.k) && (p.value = t), e.k && (_[e.k] = t);
					}
				} else t ? (_[p] = l, b(p) && (v[p] = l)) : n && (x(p, e.k) && (p.value = l), e.k && (_[e.k] = l));
			};
			if (l) {
				let t = () => {
					i(), hr.delete(e);
				};
				t.id = -1, hr.set(e, t), Li(t, r);
			} else _r(e), i();
		}
	}
}
function _r(e) {
	let t = hr.get(e);
	t && (t.flags |= 8, hr.delete(e));
}
de().requestIdleCallback, de().cancelIdleCallback;
var vr = (e) => !!e.type.__asyncLoader, yr = (e) => e.type.__isKeepAlive;
function br(e, t) {
	Sr(e, "a", t);
}
function xr(e, t) {
	Sr(e, "da", t);
}
function Sr(e, t, n = Sa) {
	let r = e.__wdc ||= () => {
		let t = n;
		for (; t;) {
			if (t.isDeactivated) return;
			t = t.parent;
		}
		return e();
	};
	if (wr(t, r, n), n) {
		let e = n.parent;
		for (; e && e.parent;) yr(e.parent.vnode) && Cr(r, t, n, e), e = e.parent;
	}
}
function Cr(e, t, n, r) {
	let i = wr(t, e, r, !0);
	jr(() => {
		c(r[t], i);
	}, n);
}
function wr(e, t, n = Sa, r = !1) {
	if (n) {
		let i = n[e] || (n[e] = []), a = t.__weh ||= (...r) => {
			We();
			let i = Ea(n), a = on(t, n, e, r);
			return i(), Ge(), a;
		};
		return r ? i.unshift(a) : i.push(a), a;
	}
}
var Tr = (e) => (t, n = Sa) => {
	(!ka || e === "sp") && wr(e, (...e) => t(...e), n);
}, Er = Tr("bm"), Dr = Tr("m"), Or = Tr("bu"), kr = Tr("u"), Ar = Tr("bum"), jr = Tr("um"), Mr = Tr("sp"), Nr = Tr("rtg"), Pr = Tr("rtc");
function Fr(e, t = Sa) {
	wr("ec", e, t);
}
var Ir = /* @__PURE__ */ Symbol.for("v-ndc");
function z(e, t, n, r) {
	let i, a = n && n[r], o = d(e);
	if (o || g(e)) {
		let n = o && /* @__PURE__ */ zt(e), r = !1, s = !1;
		n && (r = !/* @__PURE__ */ Vt(e), s = /* @__PURE__ */ Bt(e), e = rt(e)), i = Array(e.length);
		for (let n = 0, o = e.length; n < o; n++) i[n] = t(r ? s ? Gt(Wt(e[n])) : Wt(e[n]) : e[n], n, void 0, a && a[n]);
	} else if (typeof e == "number") {
		i = Array(e);
		for (let n = 0; n < e; n++) i[n] = t(n + 1, n, void 0, a && a[n]);
	} else if (v(e)) if (e[Symbol.iterator]) i = Array.from(e, (e, n) => t(e, n, void 0, a && a[n]));
	else {
		let n = Object.keys(e);
		i = Array(n.length);
		for (let r = 0, o = n.length; r < o; r++) {
			let o = n[r];
			i[r] = t(e[o], o, r, a && a[r]);
		}
	}
	else i = [];
	return n && (n[r] = i), i;
}
var Lr = (e) => e ? Oa(e) ? Ra(e) : Lr(e.parent) : null, Rr = /* @__PURE__ */ s(/* @__PURE__ */ Object.create(null), {
	$: (e) => e,
	$el: (e) => e.vnode.el,
	$data: (e) => e.data,
	$props: (e) => e.props,
	$attrs: (e) => e.attrs,
	$slots: (e) => e.slots,
	$refs: (e) => e.refs,
	$parent: (e) => Lr(e.parent),
	$root: (e) => Lr(e.root),
	$host: (e) => e.ce,
	$emit: (e) => e.emit,
	$options: (e) => qr(e),
	$forceUpdate: (e) => e.f ||= () => {
		_n(e.update);
	},
	$nextTick: (e) => e.n ||= hn.bind(e.proxy),
	$watch: (e) => Fn.bind(e)
}), zr = (e, n) => e !== t && !e.__isScriptSetup && u(e, n), Br = {
	get({ _: e }, n) {
		if (n === "__v_skip") return !0;
		let { ctx: r, setupState: i, data: a, props: o, accessCache: s, type: c, appContext: l } = e;
		if (n[0] !== "$") {
			let e = s[n];
			if (e !== void 0) switch (e) {
				case 1: return i[n];
				case 2: return a[n];
				case 4: return r[n];
				case 3: return o[n];
			}
			else if (zr(i, n)) return s[n] = 1, i[n];
			else if (a !== t && u(a, n)) return s[n] = 2, a[n];
			else if (u(o, n)) return s[n] = 3, o[n];
			else if (r !== t && u(r, n)) return s[n] = 4, r[n];
			else Hr && (s[n] = 0);
		}
		let d = Rr[n], f, p;
		if (d) return n === "$attrs" && M(e.attrs, "get", ""), d(e);
		if ((f = c.__cssModules) && (f = f[n])) return f;
		if (r !== t && u(r, n)) return s[n] = 4, r[n];
		if (p = l.config.globalProperties, u(p, n)) return p[n];
	},
	set({ _: e }, n, r) {
		let { data: i, setupState: a, ctx: o } = e;
		return zr(a, n) ? (a[n] = r, !0) : i !== t && u(i, n) ? (i[n] = r, !0) : u(e.props, n) || n[0] === "$" && n.slice(1) in e ? !1 : (o[n] = r, !0);
	},
	has({ _: { data: e, setupState: n, accessCache: r, ctx: i, appContext: a, props: o, type: s } }, c) {
		let l;
		return !!(r[c] || e !== t && c[0] !== "$" && u(e, c) || zr(n, c) || u(o, c) || u(i, c) || u(Rr, c) || u(a.config.globalProperties, c) || (l = s.__cssModules) && l[c]);
	},
	defineProperty(e, t, n) {
		return n.get == null ? u(n, "value") && this.set(e, t, n.value, null) : e._.accessCache[t] = 0, Reflect.defineProperty(e, t, n);
	}
};
function Vr(e) {
	return d(e) ? e.reduce((e, t) => (e[t] = null, e), {}) : e;
}
var Hr = !0;
function Ur(e) {
	let t = qr(e), n = e.proxy, i = e.ctx;
	Hr = !1, t.beforeCreate && Gr(t.beforeCreate, e, "bc");
	let { data: a, computed: o, methods: s, watch: c, provide: l, inject: u, created: f, beforeMount: p, mounted: m, beforeUpdate: g, updated: _, activated: y, deactivated: b, beforeDestroy: x, beforeUnmount: S, destroyed: C, unmounted: w, render: ee, renderTracked: te, renderTriggered: ne, errorCaptured: T, serverPrefetch: re, expose: E, inheritAttrs: ie, components: ae, directives: oe, filters: se } = t;
	if (u && Wr(u, i, null), s) for (let e in s) {
		let t = s[e];
		h(t) && (i[e] = t.bind(n));
	}
	if (a) {
		let t = a.call(n, n);
		v(t) && (e.data = /* @__PURE__ */ Ft(t));
	}
	if (Hr = !0, o) for (let e in o) {
		let t = o[e], a = q({
			get: h(t) ? t.bind(n, n) : h(t.get) ? t.get.bind(n, n) : r,
			set: !h(t) && h(t.set) ? t.set.bind(n) : r
		});
		Object.defineProperty(i, e, {
			enumerable: !0,
			configurable: !0,
			get: () => a.value,
			set: (e) => a.value = e
		});
	}
	if (c) for (let e in c) Kr(c[e], i, n, e);
	if (l) {
		let e = h(l) ? l.call(n) : l;
		Reflect.ownKeys(e).forEach((t) => {
			kn(t, e[t]);
		});
	}
	f && Gr(f, e, "c");
	function D(e, t) {
		d(t) ? t.forEach((t) => e(t.bind(n))) : t && e(t.bind(n));
	}
	if (D(Er, p), D(Dr, m), D(Or, g), D(kr, _), D(br, y), D(xr, b), D(Fr, T), D(Pr, te), D(Nr, ne), D(Ar, S), D(jr, w), D(Mr, re), d(E)) if (E.length) {
		let t = e.exposed ||= {};
		E.forEach((e) => {
			Object.defineProperty(t, e, {
				get: () => n[e],
				set: (t) => n[e] = t,
				enumerable: !0
			});
		});
	} else e.exposed ||= {};
	ee && e.render === r && (e.render = ee), ie != null && (e.inheritAttrs = ie), ae && (e.components = ae), oe && (e.directives = oe), re && pr(e);
}
function Wr(e, t, n = r) {
	d(e) && (e = Qr(e));
	for (let n in e) {
		let r = e[n], i;
		i = v(r) ? "default" in r ? An(r.from || n, r.default, !0) : An(r.from || n) : An(r), /* @__PURE__ */ P(i) ? Object.defineProperty(t, n, {
			enumerable: !0,
			configurable: !0,
			get: () => i.value,
			set: (e) => i.value = e
		}) : t[n] = i;
	}
}
function Gr(e, t, n) {
	on(d(e) ? e.map((e) => e.bind(t.proxy)) : e.bind(t.proxy), t, n);
}
function Kr(e, t, n, r) {
	let i = r.includes(".") ? In(n, r) : () => n[r];
	if (g(e)) {
		let n = t[e];
		h(n) && Nn(i, n);
	} else if (h(e)) Nn(i, e.bind(n));
	else if (v(e)) if (d(e)) e.forEach((e) => Kr(e, t, n, r));
	else {
		let r = h(e.handler) ? e.handler.bind(n) : t[e.handler];
		h(r) && Nn(i, r, e);
	}
}
function qr(e) {
	let t = e.type, { mixins: n, extends: r } = t, { mixins: i, optionsCache: a, config: { optionMergeStrategies: o } } = e.appContext, s = a.get(t), c;
	return s ? c = s : !i.length && !n && !r ? c = t : (c = {}, i.length && i.forEach((e) => Jr(c, e, o, !0)), Jr(c, t, o)), v(t) && a.set(t, c), c;
}
function Jr(e, t, n, r = !1) {
	let { mixins: i, extends: a } = t;
	a && Jr(e, a, n, !0), i && i.forEach((t) => Jr(e, t, n, !0));
	for (let i in t) if (!(r && i === "expose")) {
		let r = Yr[i] || n && n[i];
		e[i] = r ? r(e[i], t[i]) : t[i];
	}
	return e;
}
var Yr = {
	data: Xr,
	props: ti,
	emits: ti,
	methods: ei,
	computed: ei,
	beforeCreate: $r,
	created: $r,
	beforeMount: $r,
	mounted: $r,
	beforeUpdate: $r,
	updated: $r,
	beforeDestroy: $r,
	beforeUnmount: $r,
	destroyed: $r,
	unmounted: $r,
	activated: $r,
	deactivated: $r,
	errorCaptured: $r,
	serverPrefetch: $r,
	components: ei,
	directives: ei,
	watch: ni,
	provide: Xr,
	inject: Zr
};
function Xr(e, t) {
	return t ? e ? function() {
		return s(h(e) ? e.call(this, this) : e, h(t) ? t.call(this, this) : t);
	} : t : e;
}
function Zr(e, t) {
	return ei(Qr(e), Qr(t));
}
function Qr(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) t[e[n]] = e[n];
		return t;
	}
	return e;
}
function $r(e, t) {
	return e ? [...new Set([].concat(e, t))] : t;
}
function ei(e, t) {
	return e ? s(/* @__PURE__ */ Object.create(null), e, t) : t;
}
function ti(e, t) {
	return e ? d(e) && d(t) ? [.../* @__PURE__ */ new Set([...e, ...t])] : s(/* @__PURE__ */ Object.create(null), Vr(e), Vr(t ?? {})) : t;
}
function ni(e, t) {
	if (!e) return t;
	if (!t) return e;
	let n = s(/* @__PURE__ */ Object.create(null), e);
	for (let r in t) n[r] = $r(e[r], t[r]);
	return n;
}
function ri() {
	return {
		app: null,
		config: {
			isNativeTag: i,
			performance: !1,
			globalProperties: {},
			optionMergeStrategies: {},
			errorHandler: void 0,
			warnHandler: void 0,
			compilerOptions: {}
		},
		mixins: [],
		components: {},
		directives: {},
		provides: /* @__PURE__ */ Object.create(null),
		optionsCache: /* @__PURE__ */ new WeakMap(),
		propsCache: /* @__PURE__ */ new WeakMap(),
		emitsCache: /* @__PURE__ */ new WeakMap()
	};
}
var ii = 0;
function ai(e, t) {
	return function(n, r = null) {
		h(n) || (n = s({}, n)), r != null && !v(r) && (r = null);
		let i = ri(), a = /* @__PURE__ */ new WeakSet(), o = [], c = !1, l = i.app = {
			_uid: ii++,
			_component: n,
			_props: r,
			_container: null,
			_context: i,
			_instance: null,
			version: Va,
			get config() {
				return i.config;
			},
			set config(e) {},
			use(e, ...t) {
				return a.has(e) || (e && h(e.install) ? (a.add(e), e.install(l, ...t)) : h(e) && (a.add(e), e(l, ...t))), l;
			},
			mixin(e) {
				return i.mixins.includes(e) || i.mixins.push(e), l;
			},
			component(e, t) {
				return t ? (i.components[e] = t, l) : i.components[e];
			},
			directive(e, t) {
				return t ? (i.directives[e] = t, l) : i.directives[e];
			},
			mount(a, o, s) {
				if (!c) {
					let u = l._ceVNode || W(n, r);
					return u.appContext = i, s === !0 ? s = "svg" : s === !1 && (s = void 0), o && t ? t(u, a) : e(u, a, s), c = !0, l._container = a, a.__vue_app__ = l, Ra(u.component);
				}
			},
			onUnmount(e) {
				o.push(e);
			},
			unmount() {
				c && (on(o, l._instance, 16), e(null, l._container), delete l._container.__vue_app__);
			},
			provide(e, t) {
				return i.provides[e] = t, l;
			},
			runWithContext(e) {
				let t = oi;
				oi = l;
				try {
					return e();
				} finally {
					oi = t;
				}
			}
		};
		return l;
	};
}
var oi = null, si = (e, t) => t === "modelValue" || t === "model-value" ? e.modelModifiers : e[`${t}Modifiers`] || e[`${T(t)}Modifiers`] || e[`${E(t)}Modifiers`];
function ci(e, n, ...r) {
	if (e.isUnmounted) return;
	let i = e.vnode.props || t, a = r, o = n.startsWith("update:"), s = o && si(i, n.slice(7));
	s && (s.trim && (a = r.map((e) => g(e) ? e.trim() : e)), s.number && (a = r.map(ce)));
	let c, l = i[c = ae(n)] || i[c = ae(T(n))];
	!l && o && (l = i[c = ae(E(n))]), l && on(l, e, 6, a);
	let u = i[c + "Once"];
	if (u) {
		if (!e.emitted) e.emitted = {};
		else if (e.emitted[c]) return;
		e.emitted[c] = !0, on(u, e, 6, a);
	}
}
var li = /* @__PURE__ */ new WeakMap();
function ui(e, t, n = !1) {
	let r = n ? li : t.emitsCache, i = r.get(e);
	if (i !== void 0) return i;
	let a = e.emits, o = {}, c = !1;
	if (!h(e)) {
		let r = (e) => {
			let n = ui(e, t, !0);
			n && (c = !0, s(o, n));
		};
		!n && t.mixins.length && t.mixins.forEach(r), e.extends && r(e.extends), e.mixins && e.mixins.forEach(r);
	}
	return !a && !c ? (v(e) && r.set(e, null), null) : (d(a) ? a.forEach((e) => o[e] = null) : s(o, a), v(e) && r.set(e, o), o);
}
function di(e, t) {
	return !e || !a(t) ? !1 : (t = t.slice(2), t = t === "Once" ? t : t.replace(/Once$/, ""), u(e, t[0].toLowerCase() + t.slice(1)) || u(e, E(t)) || u(e, t));
}
function fi(e) {
	let { type: t, vnode: n, proxy: r, withProxy: i, propsOptions: [a], slots: s, attrs: c, emit: l, render: u, renderCache: d, props: f, data: p, setupState: m, ctx: h, inheritAttrs: g } = e, _ = En(e), v, y;
	try {
		if (n.shapeFlag & 4) {
			let e = i || r, t = e;
			v = ma(u.call(t, e, d, f, m, p, h)), y = c;
		} else {
			let e = t;
			v = ma(e.length > 1 ? e(f, {
				attrs: c,
				slots: s,
				emit: l
			}) : e(f, null)), y = t.props ? c : pi(c);
		}
	} catch (t) {
		$i.length = 0, sn(t, e, 1), v = W(Zi);
	}
	let b = v;
	if (y && g !== !1) {
		let e = Object.keys(y), { shapeFlag: t } = b;
		e.length && t & 7 && (a && e.some(o) && (y = mi(y, a)), b = fa(b, y, !1, !0));
	}
	return n.dirs && (b = fa(b, null, !1, !0), b.dirs = b.dirs ? b.dirs.concat(n.dirs) : n.dirs), n.transition && ur(b, n.transition), v = b, En(_), v;
}
var pi = (e) => {
	let t;
	for (let n in e) (n === "class" || n === "style" || a(n)) && ((t ||= {})[n] = e[n]);
	return t;
}, mi = (e, t) => {
	let n = {};
	for (let r in e) (!o(r) || !(r.slice(9) in t)) && (n[r] = e[r]);
	return n;
};
function hi(e, t, n) {
	let { props: r, children: i, component: a } = e, { props: o, children: s, patchFlag: c } = t, l = a.emitsOptions;
	if (t.dirs || t.transition) return !0;
	if (n && c >= 0) {
		if (c & 1024) return !0;
		if (c & 16) return r ? gi(r, o, l) : !!o;
		if (c & 8) {
			let e = t.dynamicProps;
			for (let t = 0; t < e.length; t++) {
				let n = e[t];
				if (_i(o, r, n) && !di(l, n)) return !0;
			}
		}
	} else return (i || s) && (!s || !s.$stable) ? !0 : r === o ? !1 : r ? !o || gi(r, o, l) : !!o;
	return !1;
}
function gi(e, t, n) {
	let r = Object.keys(t);
	if (r.length !== Object.keys(e).length) return !0;
	for (let i = 0; i < r.length; i++) {
		let a = r[i];
		if (_i(t, e, a) && !di(n, a)) return !0;
	}
	return !1;
}
function _i(e, t, n) {
	let r = e[n], i = t[n];
	return n === "style" && v(r) && v(i) ? !xe(r, i) : r !== i;
}
function vi({ vnode: e, parent: t, suspense: n }, r) {
	for (; t;) {
		let n = t.subTree;
		if (n.suspense && n.suspense.activeBranch === e && (n.suspense.vnode.el = n.el = r, e = n), n === e) (e = t.vnode).el = r, t = t.parent;
		else break;
	}
	n && n.activeBranch === e && (n.vnode.el = r);
}
var yi = {}, bi = () => Object.create(yi), xi = (e) => Object.getPrototypeOf(e) === yi;
function Si(e, t, n, r = !1) {
	let i = {}, a = bi();
	e.propsDefaults = /* @__PURE__ */ Object.create(null), wi(e, t, i, a);
	for (let t in e.propsOptions[0]) t in i || (i[t] = void 0);
	e.props = n ? r ? i : /* @__PURE__ */ It(i) : e.type.props ? i : a, e.attrs = a;
}
function Ci(e, t, n, r) {
	let { props: i, attrs: a, vnode: { patchFlag: o } } = e, s = /* @__PURE__ */ N(i), [c] = e.propsOptions, l = !1;
	if ((r || o > 0) && !(o & 16)) {
		if (o & 8) {
			let n = e.vnode.dynamicProps;
			for (let r = 0; r < n.length; r++) {
				let o = n[r];
				if (di(e.emitsOptions, o)) continue;
				let d = t[o];
				if (c) if (u(a, o)) d !== a[o] && (a[o] = d, l = !0);
				else {
					let t = T(o);
					i[t] = Ti(c, s, t, d, e, !1);
				}
				else d !== a[o] && (a[o] = d, l = !0);
			}
		}
	} else {
		wi(e, t, i, a) && (l = !0);
		let r;
		for (let a in s) (!t || !u(t, a) && ((r = E(a)) === a || !u(t, r))) && (c ? n && (n[a] !== void 0 || n[r] !== void 0) && (i[a] = Ti(c, s, a, void 0, e, !0)) : delete i[a]);
		if (a !== s) for (let e in a) (!t || !u(t, e)) && (delete a[e], l = !0);
	}
	l && tt(e.attrs, "set", "");
}
function wi(e, n, r, i) {
	let [a, o] = e.propsOptions, s = !1, c;
	if (n) for (let t in n) {
		if (ee(t)) continue;
		let l = n[t], d;
		a && u(a, d = T(t)) ? !o || !o.includes(d) ? r[d] = l : (c ||= {})[d] = l : di(e.emitsOptions, t) || (!(t in i) || l !== i[t]) && (i[t] = l, s = !0);
	}
	if (o) {
		let n = /* @__PURE__ */ N(r), i = c || t;
		for (let t = 0; t < o.length; t++) {
			let s = o[t];
			r[s] = Ti(a, n, s, i[s], e, !u(i, s));
		}
	}
	return s;
}
function Ti(e, t, n, r, i, a) {
	let o = e[n];
	if (o != null) {
		let e = u(o, "default");
		if (e && r === void 0) {
			let e = o.default;
			if (o.type !== Function && !o.skipFactory && h(e)) {
				let { propsDefaults: a } = i;
				if (n in a) r = a[n];
				else {
					let o = Ea(i);
					r = a[n] = e.call(null, t), o();
				}
			} else r = e;
			i.ce && i.ce._setProp(n, r);
		}
		o[0] && (a && !e ? r = !1 : o[1] && (r === "" || r === E(n)) && (r = !0));
	}
	return r;
}
var Ei = /* @__PURE__ */ new WeakMap();
function Di(e, r, i = !1) {
	let a = i ? Ei : r.propsCache, o = a.get(e);
	if (o) return o;
	let c = e.props, l = {}, f = [], p = !1;
	if (!h(e)) {
		let t = (e) => {
			p = !0;
			let [t, n] = Di(e, r, !0);
			s(l, t), n && f.push(...n);
		};
		!i && r.mixins.length && r.mixins.forEach(t), e.extends && t(e.extends), e.mixins && e.mixins.forEach(t);
	}
	if (!c && !p) return v(e) && a.set(e, n), n;
	if (d(c)) for (let e = 0; e < c.length; e++) {
		let n = T(c[e]);
		Oi(n) && (l[n] = t);
	}
	else if (c) for (let e in c) {
		let t = T(e);
		if (Oi(t)) {
			let n = c[e], r = l[t] = d(n) || h(n) ? { type: n } : s({}, n), i = r.type, a = !1, o = !0;
			if (d(i)) for (let e = 0; e < i.length; ++e) {
				let t = i[e], n = h(t) && t.name;
				if (n === "Boolean") {
					a = !0;
					break;
				}
				n === "String" && (o = !1);
			}
			else a = h(i) && i.name === "Boolean";
			r[0] = a, r[1] = o, (a || u(r, "default")) && f.push(t);
		}
	}
	let m = [l, f];
	return v(e) && a.set(e, m), m;
}
function Oi(e) {
	return e[0] !== "$" && !ee(e);
}
var ki = (e) => e === "_" || e === "_ctx" || e === "$stable", Ai = (e) => d(e) ? e.map(ma) : [ma(e)], ji = (e, t, n) => {
	if (t._n) return t;
	let r = Dn((...e) => Ai(t(...e)), n);
	return r._c = !1, r;
}, Mi = (e, t, n) => {
	let r = e._ctx;
	for (let n in e) {
		if (ki(n)) continue;
		let i = e[n];
		if (h(i)) t[n] = ji(n, i, r);
		else if (i != null) {
			let e = Ai(i);
			t[n] = () => e;
		}
	}
}, Ni = (e, t) => {
	let n = Ai(t);
	e.slots.default = () => n;
}, Pi = (e, t, n) => {
	for (let r in t) (n || !ki(r)) && (e[r] = t[r]);
}, Fi = (e, t, n) => {
	let r = e.slots = bi();
	if (e.vnode.shapeFlag & 32) {
		let e = t._;
		e ? (Pi(r, t, n), n && D(r, "_", e, !0)) : Mi(t, r);
	} else t && Ni(e, t);
}, Ii = (e, n, r) => {
	let { vnode: i, slots: a } = e, o = !0, s = t;
	if (i.shapeFlag & 32) {
		let e = n._;
		e ? r && e === 1 ? o = !1 : Pi(a, n, r) : (o = !n.$stable, Mi(n, a)), s = n;
	} else n && (Ni(e, n), s = { default: 1 });
	if (o) for (let e in a) !ki(e) && s[e] == null && delete a[e];
}, Li = Yi;
function Ri(e) {
	return zi(e);
}
function zi(e, i) {
	let a = de();
	a.__VUE__ = !0;
	let { insert: o, remove: s, patchProp: c, createElement: l, createText: u, createComment: d, setText: f, setElementText: p, parentNode: m, nextSibling: h, setScopeId: g = r, insertStaticContent: _ } = e, v = (e, t, n, r = null, i = null, a = null, o = void 0, s = null, c = !!t.dynamicChildren) => {
		if (e === t) return;
		e && !sa(e, t) && (r = be(e), ge(e, i, a, !0), e = null), t.patchFlag === -2 && (c = !1, t.dynamicChildren = null);
		let { type: l, ref: u, shapeFlag: d } = t;
		switch (l) {
			case Xi:
				y(e, t, n, r);
				break;
			case Zi:
				b(e, t, n, r);
				break;
			case Qi:
				e ?? x(t, n, r, o);
				break;
			case B:
				ae(e, t, n, r, i, a, o, s, c);
				break;
			default: d & 1 ? w(e, t, n, r, i, a, o, s, c) : d & 6 ? oe(e, t, n, r, i, a, o, s, c) : (d & 64 || d & 128) && l.process(e, t, n, r, i, a, o, s, c, Ce);
		}
		u != null && i ? gr(u, e && e.ref, a, t || e, !t) : u == null && e && e.ref != null && gr(e.ref, null, a, e, !0);
	}, y = (e, t, n, r) => {
		if (e == null) o(t.el = u(t.children), n, r);
		else {
			let n = t.el = e.el;
			t.children !== e.children && f(n, t.children);
		}
	}, b = (e, t, n, r) => {
		e == null ? o(t.el = d(t.children || ""), n, r) : t.el = e.el;
	}, x = (e, t, n, r) => {
		[e.el, e.anchor] = _(e.children, t, n, r, e.el, e.anchor);
	}, S = ({ el: e, anchor: t }, n, r) => {
		let i;
		for (; e && e !== t;) i = h(e), o(e, n, r), e = i;
		o(t, n, r);
	}, C = ({ el: e, anchor: t }) => {
		let n;
		for (; e && e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, w = (e, t, n, r, i, a, o, s, c) => {
		if (t.type === "svg" ? o = "svg" : t.type === "math" && (o = "mathml"), e == null) te(t, n, r, i, a, o, s, c);
		else {
			let n = e.el && e.el._isVueCE ? e.el : null;
			try {
				n && n._beginPatch(), re(e, t, i, a, o, s, c);
			} finally {
				n && n._endPatch();
			}
		}
	}, te = (e, t, n, r, i, a, s, u) => {
		let d, f, { props: m, shapeFlag: h, transition: g, dirs: _ } = e;
		if (d = e.el = l(e.type, a, m && m.is, m), h & 8 ? p(d, e.children) : h & 16 && T(e.children, d, null, r, i, Bi(e, a), s, u), _ && On(e, null, r, "created"), ne(d, e, e.scopeId, s, r), m) {
			for (let e in m) e !== "value" && !ee(e) && c(d, e, null, m[e], a, r);
			"value" in m && c(d, "value", null, m.value, a), (f = m.onVnodeBeforeMount) && va(f, r, e);
		}
		_ && On(e, null, r, "beforeMount");
		let v = Hi(i, g);
		v && g.beforeEnter(d), o(d, t, n), ((f = m && m.onVnodeMounted) || v || _) && Li(() => {
			try {
				f && va(f, r, e), v && g.enter(d), _ && On(e, null, r, "mounted");
			} finally {}
		}, i);
	}, ne = (e, t, n, r, i) => {
		if (n && g(e, n), r) for (let t = 0; t < r.length; t++) g(e, r[t]);
		if (i) {
			let n = i.subTree;
			if (t === n || Ji(n.type) && (n.ssContent === t || n.ssFallback === t)) {
				let t = i.vnode;
				ne(e, t, t.scopeId, t.slotScopeIds, i.parent);
			}
		}
	}, T = (e, t, n, r, i, a, o, s, c = 0) => {
		for (let l = c; l < e.length; l++) {
			let c = e[l] = s ? ha(e[l]) : ma(e[l]);
			v(null, c, t, n, r, i, a, o, s);
		}
	}, re = (e, n, r, i, a, o, s) => {
		let l = n.el = e.el, { patchFlag: u, dynamicChildren: d, dirs: f } = n;
		u |= e.patchFlag & 16;
		let m = e.props || t, h = n.props || t, g;
		if (r && Vi(r, !1), (g = h.onVnodeBeforeUpdate) && va(g, r, n, e), f && On(n, e, r, "beforeUpdate"), r && Vi(r, !0), d && (!e.dynamicChildren || e.dynamicChildren.length !== d.length) && (u = 0, s = !1, d = null), (m.innerHTML && h.innerHTML == null || m.textContent && h.textContent == null) && p(l, ""), d ? E(e.dynamicChildren, d, l, r, i, Bi(n, a), o) : s || fe(e, n, l, null, r, i, Bi(n, a), o, !1), u > 0) {
			if (u & 16) ie(l, m, h, r, a);
			else if (u & 2 && m.class !== h.class && c(l, "class", null, h.class, a), u & 4 && c(l, "style", m.style, h.style, a), u & 8) {
				let e = n.dynamicProps;
				for (let t = 0; t < e.length; t++) {
					let n = e[t], i = m[n], o = h[n];
					(o !== i || n === "value") && c(l, n, i, o, a, r);
				}
			}
			u & 1 && e.children !== n.children && p(l, n.children);
		} else !s && d == null && ie(l, m, h, r, a);
		((g = h.onVnodeUpdated) || f) && Li(() => {
			g && va(g, r, n, e), f && On(n, e, r, "updated");
		}, i);
	}, E = (e, t, n, r, i, a, o) => {
		for (let s = 0; s < t.length; s++) {
			let c = e[s], l = t[s], u = c.el && (c.type === B || !sa(c, l) || c.shapeFlag & 198) ? m(c.el) : n;
			v(c, l, u, null, r, i, a, o, !0);
		}
	}, ie = (e, n, r, i, a) => {
		if (n !== r) {
			if (n !== t) for (let t in n) !ee(t) && !(t in r) && c(e, t, n[t], null, a, i);
			for (let t in r) {
				if (ee(t)) continue;
				let o = r[t], s = n[t];
				o !== s && t !== "value" && c(e, t, s, o, a, i);
			}
			"value" in r && c(e, "value", n.value, r.value, a);
		}
	}, ae = (e, t, n, r, i, a, s, c, l) => {
		let d = t.el = e ? e.el : u(""), f = t.anchor = e ? e.anchor : u(""), { patchFlag: p, dynamicChildren: m, slotScopeIds: h } = t;
		h && (c = c ? c.concat(h) : h), e == null ? (o(d, n, r), o(f, n, r), T(t.children || [], n, f, i, a, s, c, l)) : p > 0 && p & 64 && m && e.dynamicChildren && e.dynamicChildren.length === m.length ? (E(e.dynamicChildren, m, n, i, a, s, c), (t.key != null || i && t === i.subTree) && Ui(e, t, !0)) : fe(e, t, n, f, i, a, s, c, l);
	}, oe = (e, t, n, r, i, a, o, s, c) => {
		t.slotScopeIds = s, e == null ? t.shapeFlag & 512 ? i.ctx.activate(t, n, r, o, c) : D(t, n, r, i, a, o, c) : ce(e, t, c);
	}, D = (e, t, n, r, i, a, o) => {
		let s = e.component = xa(e, r, i);
		if (yr(e) && (s.ctx.renderer = Ce), Aa(s, !1, o), s.asyncDep) {
			if (i && i.registerDep(s, le, o), !e.el) {
				let r = s.subTree = W(Zi);
				b(null, r, t, n), e.placeholder = r.el;
			}
		} else le(s, e, t, n, i, a, o);
	}, ce = (e, t, n) => {
		let r = t.component = e.component;
		if (hi(e, t, n)) if (r.asyncDep && !r.asyncResolved) {
			ue(r, t, n);
			return;
		} else r.next = t, r.update();
		else t.el = e.el, r.vnode = t;
	}, le = (e, t, n, r, i, a, o) => {
		let s = () => {
			if (e.isMounted) {
				let { next: t, bu: n, u: r, parent: s, vnode: c } = e;
				{
					let n = Gi(e);
					if (n) {
						t && (t.el = c.el, ue(e, t, o)), n.asyncDep.then(() => {
							Li(() => {
								e.isUnmounted || l();
							}, i);
						});
						return;
					}
				}
				let u = t, d;
				Vi(e, !1), t ? (t.el = c.el, ue(e, t, o)) : t = c, n && se(n), (d = t.props && t.props.onVnodeBeforeUpdate) && va(d, s, t, c), Vi(e, !0);
				let f = fi(e), p = e.subTree;
				e.subTree = f, v(p, f, m(p.el), be(p), e, i, a), t.el = f.el, u === null && vi(e, f.el), r && Li(r, i), (d = t.props && t.props.onVnodeUpdated) && Li(() => va(d, s, t, c), i);
			} else {
				let o, { el: s, props: c } = t, { bm: l, m: u, parent: d, root: f, type: p } = e, m = vr(t);
				if (Vi(e, !1), l && se(l), !m && (o = c && c.onVnodeBeforeMount) && va(o, d, t), Vi(e, !0), s && we) {
					let t = () => {
						e.subTree = fi(e), we(s, e.subTree, e, i, null);
					};
					m && p.__asyncHydrate ? p.__asyncHydrate(s, e, t) : t();
				} else {
					f.ce && f.ce._hasShadowRoot() && f.ce._injectChildStyle(p, e.parent ? e.parent.type : void 0);
					let o = e.subTree = fi(e);
					v(null, o, n, r, e, i, a), t.el = o.el;
				}
				if (u && Li(u, i), !m && (o = c && c.onVnodeMounted)) {
					let e = t;
					Li(() => va(o, d, e), i);
				}
				(t.shapeFlag & 256 || d && vr(d.vnode) && d.vnode.shapeFlag & 256) && e.a && Li(e.a, i), e.isMounted = !0, t = n = r = null;
			}
		};
		e.scope.on();
		let c = e.effect = new ke(s);
		e.scope.off();
		let l = e.update = c.run.bind(c), u = e.job = c.runIfDirty.bind(c);
		u.i = e, u.id = e.uid, c.scheduler = () => _n(u), Vi(e, !0), l();
	}, ue = (e, t, n) => {
		t.component = e;
		let r = e.vnode.props;
		e.vnode = t, e.next = null, Ci(e, t.props, r, n), Ii(e, t.children, n), We(), bn(e), Ge();
	}, fe = (e, t, n, r, i, a, o, s, c = !1) => {
		let l = e && e.children, u = e ? e.shapeFlag : 0, d = t.children, { patchFlag: f, shapeFlag: m } = t;
		if (f > 0) {
			if (f & 128) {
				me(l, d, n, r, i, a, o, s, c);
				return;
			}
			if (f & 256) {
				pe(l, d, n, r, i, a, o, s, c);
				return;
			}
		}
		m & 8 ? (u & 16 && ye(l, i, a), d !== l && p(n, d)) : u & 16 ? m & 16 ? me(l, d, n, r, i, a, o, s, c) : ye(l, i, a, !0) : (u & 8 && p(n, ""), m & 16 && T(d, n, r, i, a, o, s, c));
	}, pe = (e, t, r, i, a, o, s, c, l) => {
		e ||= n, t ||= n;
		let u = e.length, d = t.length, f = Math.min(u, d), p;
		for (p = 0; p < f; p++) {
			let n = t[p] = l ? ha(t[p]) : ma(t[p]);
			v(e[p], n, r, null, a, o, s, c, l);
		}
		u > d ? ye(e, a, o, !0, !1, f) : T(t, r, i, a, o, s, c, l, f);
	}, me = (e, t, r, i, a, o, s, c, l) => {
		let u = 0, d = t.length, f = e.length - 1, p = d - 1;
		for (; u <= f && u <= p;) {
			let n = e[u], i = t[u] = l ? ha(t[u]) : ma(t[u]);
			if (sa(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			u++;
		}
		for (; u <= f && u <= p;) {
			let n = e[f], i = t[p] = l ? ha(t[p]) : ma(t[p]);
			if (sa(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			f--, p--;
		}
		if (u > f) {
			if (u <= p) {
				let e = p + 1, n = e < d ? t[e].el : i;
				for (; u <= p;) v(null, t[u] = l ? ha(t[u]) : ma(t[u]), r, n, a, o, s, c, l), u++;
			}
		} else if (u > p) for (; u <= f;) ge(e[u], a, o, !0), u++;
		else {
			let m = u, h = u, g = /* @__PURE__ */ new Map();
			for (u = h; u <= p; u++) {
				let e = t[u] = l ? ha(t[u]) : ma(t[u]);
				e.key != null && g.set(e.key, u);
			}
			let _, y = 0, b = p - h + 1, x = !1, S = 0, C = Array(b);
			for (u = 0; u < b; u++) C[u] = 0;
			for (u = m; u <= f; u++) {
				let n = e[u];
				if (y >= b) {
					ge(n, a, o, !0);
					continue;
				}
				let i;
				if (n.key != null) i = g.get(n.key);
				else for (_ = h; _ <= p; _++) if (C[_ - h] === 0 && sa(n, t[_])) {
					i = _;
					break;
				}
				i === void 0 ? ge(n, a, o, !0) : (C[i - h] = u + 1, i >= S ? S = i : x = !0, v(n, t[i], r, null, a, o, s, c, l), y++);
			}
			let w = x ? Wi(C) : n;
			for (_ = w.length - 1, u = b - 1; u >= 0; u--) {
				let e = h + u, n = t[e], f = t[e + 1], p = e + 1 < d ? f.el || qi(f) : i;
				C[u] === 0 ? v(null, n, r, p, a, o, s, c, l) : x && (_ < 0 || u !== w[_] ? he(n, r, p, 2) : _--);
			}
		}
	}, he = (e, t, n, r, i = null) => {
		let { el: a, type: c, transition: l, children: u, shapeFlag: d } = e;
		if (d & 6) {
			he(e.component.subTree, t, n, r);
			return;
		}
		if (d & 128) {
			e.suspense.move(t, n, r);
			return;
		}
		if (d & 64) {
			c.move(e, t, n, Ce);
			return;
		}
		if (c === B) {
			o(a, t, n);
			for (let e = 0; e < u.length; e++) he(u[e], t, n, r);
			o(e.anchor, t, n);
			return;
		}
		if (c === Qi) {
			S(e, t, n);
			return;
		}
		if (r !== 2 && d & 1 && l) if (r === 0) l.persisted && !a[Zn] ? o(a, t, n) : (l.beforeEnter(a), o(a, t, n), Li(() => l.enter(a), i));
		else {
			let { leave: r, delayLeave: i, afterLeave: c } = l, u = () => {
				e.ctx.isUnmounted ? s(a) : o(a, t, n);
			}, d = () => {
				let e = a._isLeaving || !!a[Zn];
				a._isLeaving && a[Zn](!0), l.persisted && !e ? u() : r(a, () => {
					u(), c && c();
				});
			};
			i ? i(a, u, d) : d();
		}
		else o(a, t, n);
	}, ge = (e, t, n, r = !1, i = !1) => {
		let { type: a, props: o, ref: s, children: c, dynamicChildren: l, shapeFlag: u, patchFlag: d, dirs: f, cacheIndex: p, memo: m } = e;
		if (d === -2 && (i = !1), s != null && (We(), gr(s, null, n, e, !0), Ge()), p != null && (t.renderCache[p] = void 0), u & 256) {
			t.ctx.deactivate(e);
			return;
		}
		let h = u & 1 && f, g = !vr(e), _;
		if (g && (_ = o && o.onVnodeBeforeUnmount) && va(_, t, e), u & 6) ve(e.component, n, r);
		else {
			if (u & 128) {
				e.suspense.unmount(n, r);
				return;
			}
			h && On(e, null, t, "beforeUnmount"), u & 64 ? e.type.remove(e, t, n, Ce, r) : l && !l.hasOnce && (a !== B || d > 0 && d & 64) ? ye(l, t, n, !1, !0) : (a === B && d & 384 || !i && u & 16) && ye(c, t, n), r && O(e);
		}
		let v = m != null && p == null;
		(g && (_ = o && o.onVnodeUnmounted) || h || v) && Li(() => {
			_ && va(_, t, e), h && On(e, null, t, "unmounted"), v && (e.el = null);
		}, n);
	}, O = (e) => {
		let { type: t, el: n, anchor: r, transition: i } = e;
		if (t === B) {
			_e(n, r);
			return;
		}
		if (t === Qi) {
			C(e);
			return;
		}
		let a = () => {
			s(n), i && !i.persisted && i.afterLeave && i.afterLeave();
		};
		if (e.shapeFlag & 1 && i && !i.persisted) {
			let { leave: t, delayLeave: r } = i, o = () => t(n, a);
			r ? r(e.el, a, o) : o();
		} else a();
	}, _e = (e, t) => {
		let n;
		for (; e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, ve = (e, t, n) => {
		let { bum: r, scope: i, job: a, subTree: o, um: s, m: c, a: l } = e;
		Ki(c), Ki(l), r && se(r), i.stop(), a && (a.flags |= 8, ge(o, e, t, n)), s && Li(s, t), Li(() => {
			e.isUnmounted = !0;
		}, t);
	}, ye = (e, t, n, r = !1, i = !1, a = 0) => {
		for (let o = a; o < e.length; o++) ge(e[o], t, n, r, i);
	}, be = (e) => {
		if (e.shapeFlag & 6) return be(e.component.subTree);
		if (e.shapeFlag & 128) return e.suspense.next();
		let t = h(e.anchor || e.el), n = t && t[Rn];
		return n ? h(n) : t;
	}, xe = !1, Se = (e, t, n) => {
		let r;
		e == null ? t._vnode && (ge(t._vnode, null, null, !0), r = t._vnode.component) : v(t._vnode || null, e, t, null, null, null, n), t._vnode = e, xe ||= (xe = !0, bn(r), xn(), !1);
	}, Ce = {
		p: v,
		um: ge,
		m: he,
		r: O,
		mt: D,
		mc: T,
		pc: fe,
		pbc: E,
		n: be,
		o: e
	}, k, we;
	return i && ([k, we] = i(Ce)), {
		render: Se,
		hydrate: k,
		createApp: ai(Se, k)
	};
}
function Bi({ type: e, props: t }, n) {
	return n === "svg" && e === "foreignObject" || n === "mathml" && e === "annotation-xml" && t && t.encoding && t.encoding.includes("html") ? void 0 : n;
}
function Vi({ effect: e, job: t }, n) {
	n ? (e.flags |= 32, t.flags |= 4) : (e.flags &= -33, t.flags &= -5);
}
function Hi(e, t) {
	return (!e || e && !e.pendingBranch) && t && !t.persisted;
}
function Ui(e, t, n = !1) {
	let r = e.children, i = t.children;
	if (d(r) && d(i)) for (let e = 0; e < r.length; e++) {
		let t = r[e], a = i[e];
		a.shapeFlag & 1 && !a.dynamicChildren && ((a.patchFlag <= 0 || a.patchFlag === 32) && (a = i[e] = ha(i[e]), a.el = t.el), !n && a.patchFlag !== -2 && Ui(t, a)), a.type === Xi && (a.patchFlag === -1 && (a = i[e] = ha(a)), a.el = t.el), a.type === Zi && !a.el && (a.el = t.el);
	}
}
function Wi(e) {
	let t = e.slice(), n = [0], r, i, a, o, s, c = e.length;
	for (r = 0; r < c; r++) {
		let c = e[r];
		if (c !== 0) {
			if (i = n[n.length - 1], e[i] < c) {
				t[r] = i, n.push(r);
				continue;
			}
			for (a = 0, o = n.length - 1; a < o;) s = a + o >> 1, e[n[s]] < c ? a = s + 1 : o = s;
			c < e[n[a]] && (a > 0 && (t[r] = n[a - 1]), n[a] = r);
		}
	}
	for (a = n.length, o = n[a - 1]; a-- > 0;) n[a] = o, o = t[o];
	return n;
}
function Gi(e) {
	let t = e.subTree.component;
	if (t) return t.asyncDep && !t.asyncResolved ? t : Gi(t);
}
function Ki(e) {
	if (e) for (let t = 0; t < e.length; t++) e[t].flags |= 8;
}
function qi(e) {
	if (e.placeholder) return e.placeholder;
	let t = e.component;
	return t ? qi(t.subTree) : null;
}
var Ji = (e) => e.__isSuspense;
function Yi(e, t) {
	t && t.pendingBranch ? d(e) ? t.effects.push(...e) : t.effects.push(e) : yn(e);
}
var B = /* @__PURE__ */ Symbol.for("v-fgt"), Xi = /* @__PURE__ */ Symbol.for("v-txt"), Zi = /* @__PURE__ */ Symbol.for("v-cmt"), Qi = /* @__PURE__ */ Symbol.for("v-stc"), $i = [], ea = null;
function V(e = !1) {
	$i.push(ea = e ? null : []);
}
function ta() {
	$i.pop(), ea = $i[$i.length - 1] || null;
}
var na = 1;
function ra(e, t = !1) {
	na += e, e < 0 && ea && t && (ea.hasOnce = !0);
}
function ia(e) {
	return e.dynamicChildren = na > 0 ? ea || n : null, ta(), na > 0 && ea && ea.push(e), e;
}
function H(e, t, n, r, i, a) {
	return ia(U(e, t, n, r, i, a, !0));
}
function aa(e, t, n, r, i) {
	return ia(W(e, t, n, r, i, !0));
}
function oa(e) {
	return e ? e.__v_isVNode === !0 : !1;
}
function sa(e, t) {
	return e.type === t.type && e.key === t.key;
}
var ca = ({ key: e }) => e ?? null, la = ({ ref: e, ref_key: t, ref_for: n }) => (typeof e == "number" && (e = "" + e), e == null ? null : g(e) || /* @__PURE__ */ P(e) || h(e) ? {
	i: wn,
	r: e,
	k: t,
	f: !!n
} : e);
function U(e, t = null, n = null, r = 0, i = null, a = e === B ? 0 : 1, o = !1, s = !1) {
	let c = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e,
		props: t,
		key: t && ca(t),
		ref: t && la(t),
		scopeId: Tn,
		slotScopeIds: null,
		children: n,
		component: null,
		suspense: null,
		ssContent: null,
		ssFallback: null,
		dirs: null,
		transition: null,
		el: null,
		anchor: null,
		target: null,
		targetStart: null,
		targetAnchor: null,
		staticCount: 0,
		shapeFlag: a,
		patchFlag: r,
		dynamicProps: i,
		dynamicChildren: null,
		appContext: null,
		ctx: wn
	};
	return s ? (ga(c, n), a & 128 && e.normalize(c)) : n && (c.shapeFlag |= g(n) ? 8 : 16), na > 0 && !o && ea && (c.patchFlag > 0 || a & 6) && c.patchFlag !== 32 && ea.push(c), c;
}
var W = ua;
function ua(e, t = null, n = null, r = 0, i = null, a = !1) {
	if ((!e || e === Ir) && (e = Zi), oa(e)) {
		let r = fa(e, t, !0);
		return n && ga(r, n), na > 0 && !a && ea && (r.shapeFlag & 6 ? ea[ea.indexOf(e)] = r : ea.push(r)), r.patchFlag = -2, r;
	}
	if (za(e) && (e = e.__vccOpts), t) {
		t = da(t);
		let { class: e, style: n } = t;
		e && !g(e) && (t.class = O(e)), v(n) && (/* @__PURE__ */ Ht(n) && !d(n) && (n = s({}, n)), t.style = fe(n));
	}
	let o = g(e) ? 1 : Ji(e) ? 128 : zn(e) ? 64 : v(e) ? 4 : h(e) ? 2 : 0;
	return U(e, t, n, r, i, o, a, !0);
}
function da(e) {
	return e ? /* @__PURE__ */ Ht(e) || xi(e) ? s({}, e) : e : null;
}
function fa(e, t, n = !1, r = !1) {
	let { props: i, ref: a, patchFlag: o, children: s, transition: c } = e, l = t ? _a(i || {}, t) : i, u = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e.type,
		props: l,
		key: l && ca(l),
		ref: t && t.ref ? n && a ? d(a) ? a.concat(la(t)) : [a, la(t)] : la(t) : a,
		scopeId: e.scopeId,
		slotScopeIds: e.slotScopeIds,
		children: s,
		target: e.target,
		targetStart: e.targetStart,
		targetAnchor: e.targetAnchor,
		staticCount: e.staticCount,
		shapeFlag: e.shapeFlag,
		patchFlag: t && e.type !== B ? o === -1 ? 16 : o | 16 : o,
		dynamicProps: e.dynamicProps,
		dynamicChildren: e.dynamicChildren,
		appContext: e.appContext,
		dirs: e.dirs,
		transition: c,
		component: e.component,
		suspense: e.suspense,
		ssContent: e.ssContent && fa(e.ssContent),
		ssFallback: e.ssFallback && fa(e.ssFallback),
		placeholder: e.placeholder,
		el: e.el,
		anchor: e.anchor,
		ctx: e.ctx,
		ce: e.ce
	};
	return c && r && ur(u, c.clone(u)), u;
}
function G(e = " ", t = 0) {
	return W(Xi, null, e, t);
}
function pa(e, t) {
	let n = W(Qi, null, e);
	return n.staticCount = t, n;
}
function K(e = "", t = !1) {
	return t ? (V(), aa(Zi, null, e)) : W(Zi, null, e);
}
function ma(e) {
	return e == null || typeof e == "boolean" ? W(Zi) : d(e) ? W(B, null, e.slice()) : oa(e) ? ha(e) : W(Xi, null, String(e));
}
function ha(e) {
	return e.el === null && e.patchFlag !== -1 || e.memo ? e : fa(e);
}
function ga(e, t) {
	let n = 0, { shapeFlag: r } = e;
	if (t == null) t = null;
	else if (d(t)) n = 16;
	else if (typeof t == "object") if (r & 65) {
		let n = t.default;
		n && (n._c && (n._d = !1), ga(e, n()), n._c && (n._d = !0));
		return;
	} else {
		n = 32;
		let r = t._;
		!r && !xi(t) ? t._ctx = wn : r === 3 && wn && (wn.slots._ === 1 ? t._ = 1 : (t._ = 2, e.patchFlag |= 1024));
	}
	else if (h(t)) {
		if (r & 65) {
			ga(e, { default: t });
			return;
		}
		t = {
			default: t,
			_ctx: wn
		}, n = 32;
	} else t = String(t), r & 64 ? (n = 16, t = [G(t)]) : n = 8;
	e.children = t, e.shapeFlag |= n;
}
function _a(...e) {
	let t = {};
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		for (let e in r) if (e === "class") t.class !== r.class && (t.class = O([t.class, r.class]));
		else if (e === "style") t.style = fe([t.style, r.style]);
		else if (a(e)) {
			let n = t[e], i = r[e];
			i && n !== i && !(d(n) && n.includes(i)) ? t[e] = n ? [].concat(n, i) : i : i == null && n == null && !o(e) && (t[e] = i);
		} else e !== "" && (t[e] = r[e]);
	}
	return t;
}
function va(e, t, n, r = null) {
	on(e, t, 7, [n, r]);
}
var ya = ri(), ba = 0;
function xa(e, n, r) {
	let i = e.type, a = (n ? n.appContext : e.appContext) || ya, o = {
		uid: ba++,
		vnode: e,
		type: i,
		parent: n,
		appContext: a,
		root: null,
		next: null,
		subTree: null,
		effect: null,
		update: null,
		job: null,
		scope: new Ee(!0),
		render: null,
		proxy: null,
		exposed: null,
		exposeProxy: null,
		withProxy: null,
		provides: n ? n.provides : Object.create(a.provides),
		ids: n ? n.ids : [
			"",
			0,
			0
		],
		accessCache: null,
		renderCache: [],
		components: null,
		directives: null,
		propsOptions: Di(i, a),
		emitsOptions: ui(i, a),
		emit: null,
		emitted: null,
		propsDefaults: t,
		inheritAttrs: i.inheritAttrs,
		ctx: t,
		data: t,
		props: t,
		attrs: t,
		slots: t,
		refs: t,
		setupState: t,
		setupContext: null,
		suspense: r,
		suspenseId: r ? r.pendingId : 0,
		asyncDep: null,
		asyncResolved: !1,
		isMounted: !1,
		isUnmounted: !1,
		isDeactivated: !1,
		bc: null,
		c: null,
		bm: null,
		m: null,
		bu: null,
		u: null,
		um: null,
		bum: null,
		da: null,
		a: null,
		rtg: null,
		rtc: null,
		ec: null,
		sp: null
	};
	return o.ctx = { _: o }, o.root = n ? n.root : o, o.emit = ci.bind(null, o), e.ce && e.ce(o), o;
}
var Sa = null, Ca = () => Sa || wn, wa, Ta;
{
	let e = de(), t = (t, n) => {
		let r;
		return (r = e[t]) || (r = e[t] = []), r.push(n), (e) => {
			r.length > 1 ? r.forEach((t) => t(e)) : r[0](e);
		};
	};
	wa = t("__VUE_INSTANCE_SETTERS__", (e) => Sa = e), Ta = t("__VUE_SSR_SETTERS__", (e) => ka = e);
}
var Ea = (e) => {
	let t = Sa;
	return wa(e), e.scope.on(), () => {
		e.scope.off(), wa(t);
	};
}, Da = () => {
	Sa && Sa.scope.off(), wa(null);
};
function Oa(e) {
	return e.vnode.shapeFlag & 4;
}
var ka = !1;
function Aa(e, t = !1, n = !1) {
	t && Ta(t);
	let { props: r, children: i } = e.vnode, a = Oa(e);
	Si(e, r, a, t), Fi(e, i, n || t);
	let o = a ? ja(e, t) : void 0;
	return t && Ta(!1), o;
}
function ja(e, t) {
	let n = e.type;
	e.accessCache = /* @__PURE__ */ Object.create(null), e.proxy = new Proxy(e.ctx, Br);
	let { setup: r } = n;
	if (r) {
		We();
		let n = e.setupContext = r.length > 1 ? La(e) : null, i = Ea(e), a = an(r, e, 0, [e.props, n]), o = y(a);
		if (Ge(), i(), (o || e.sp) && !vr(e) && pr(e), o) {
			if (a.then(Da, Da), t) return a.then((n) => {
				Ma(e, n, t);
			}).catch((t) => {
				sn(t, e, 0);
			});
			e.asyncDep = a;
		} else Ma(e, a, t);
	} else Fa(e, t);
}
function Ma(e, t, n) {
	h(t) ? e.type.__ssrInlineRender ? e.ssrRender = t : e.render = t : v(t) && (e.setupState = Yt(t)), Fa(e, n);
}
var Na, Pa;
function Fa(e, t, n) {
	let i = e.type;
	if (!e.render) {
		if (!t && Na && !i.render) {
			let t = i.template || qr(e).template;
			if (t) {
				let { isCustomElement: n, compilerOptions: r } = e.appContext.config, { delimiters: a, compilerOptions: o } = i;
				i.render = Na(t, s(s({
					isCustomElement: n,
					delimiters: a
				}, r), o));
			}
		}
		e.render = i.render || r, Pa && Pa(e);
	}
	{
		let t = Ea(e);
		We();
		try {
			Ur(e);
		} finally {
			Ge(), t();
		}
	}
}
var Ia = { get(e, t) {
	return M(e, "get", ""), e[t];
} };
function La(e) {
	return {
		attrs: new Proxy(e.attrs, Ia),
		slots: e.slots,
		emit: e.emit,
		expose: (t) => {
			e.exposed = t || {};
		}
	};
}
function Ra(e) {
	return e.exposed ? e.exposeProxy ||= new Proxy(Yt(Ut(e.exposed)), {
		get(t, n) {
			if (n in t) return t[n];
			if (n in Rr) return Rr[n](e);
		},
		has(e, t) {
			return t in e || t in Rr;
		}
	}) : e.proxy;
}
function za(e) {
	return h(e) && "__vccOpts" in e;
}
var q = (e, t) => /* @__PURE__ */ Zt(e, t, ka);
function Ba(e, t, n) {
	try {
		ra(-1);
		let r = arguments.length;
		return r === 2 ? v(t) && !d(t) ? oa(t) ? W(e, null, [t]) : W(e, t) : W(e, null, t) : (r > 3 ? n = Array.prototype.slice.call(arguments, 2) : r === 3 && oa(n) && (n = [n]), W(e, t, n));
	} finally {
		ra(1);
	}
}
var Va = "3.5.40", Ha = void 0, Ua = typeof window < "u" && window.trustedTypes;
if (Ua) try {
	Ha = /* @__PURE__ */ Ua.createPolicy("vue", { createHTML: (e) => e });
} catch {}
var Wa = Ha ? (e) => Ha.createHTML(e) : (e) => e, Ga = "http://www.w3.org/2000/svg", Ka = "http://www.w3.org/1998/Math/MathML", qa = typeof document < "u" ? document : null, Ja = qa && /* @__PURE__ */ qa.createElement("template"), Ya = {
	insert: (e, t, n) => {
		t.insertBefore(e, n || null);
	},
	remove: (e) => {
		let t = e.parentNode;
		t && t.removeChild(e);
	},
	createElement: (e, t, n, r) => {
		let i = t === "svg" ? qa.createElementNS(Ga, e) : t === "mathml" ? qa.createElementNS(Ka, e) : n ? qa.createElement(e, { is: n }) : qa.createElement(e);
		return e === "select" && r && r.multiple != null && i.setAttribute("multiple", r.multiple), i;
	},
	createText: (e) => qa.createTextNode(e),
	createComment: (e) => qa.createComment(e),
	setText: (e, t) => {
		e.nodeValue = t;
	},
	setElementText: (e, t) => {
		e.textContent = t;
	},
	parentNode: (e) => e.parentNode,
	nextSibling: (e) => e.nextSibling,
	querySelector: (e) => qa.querySelector(e),
	setScopeId(e, t) {
		e.setAttribute(t, "");
	},
	insertStaticContent(e, t, n, r, i, a) {
		let o = n ? n.previousSibling : t.lastChild;
		if (i && (i === a || i.nextSibling)) for (; t.insertBefore(i.cloneNode(!0), n), !(i === a || !(i = i.nextSibling)););
		else {
			Ja.innerHTML = Wa(r === "svg" ? `<svg>${e}</svg>` : r === "mathml" ? `<math>${e}</math>` : e);
			let i = Ja.content;
			if (r === "svg" || r === "mathml") {
				let e = i.firstChild;
				for (; e.firstChild;) i.appendChild(e.firstChild);
				i.removeChild(e);
			}
			t.insertBefore(i, n);
		}
		return [o ? o.nextSibling : t.firstChild, n ? n.previousSibling : t.lastChild];
	}
}, Xa = "transition", Za = "animation", Qa = /* @__PURE__ */ Symbol("_vtc"), $a = {
	name: String,
	type: String,
	css: {
		type: Boolean,
		default: !0
	},
	duration: [
		String,
		Number,
		Object
	],
	enterFromClass: String,
	enterActiveClass: String,
	enterToClass: String,
	appearFromClass: String,
	appearActiveClass: String,
	appearToClass: String,
	leaveFromClass: String,
	leaveActiveClass: String,
	leaveToClass: String
}, eo = /* @__PURE__ */ s({}, tr, $a), to = /* @__PURE__ */ ((e) => (e.displayName = "Transition", e.props = eo, e))((e, { slots: t }) => Ba(ar, io(e), t)), no = (e, t = []) => {
	d(e) ? e.forEach((e) => e(...t)) : e && e(...t);
}, ro = (e) => e ? d(e) ? e.some((e) => e.length > 1) : e.length > 1 : !1;
function io(e) {
	let t = {};
	for (let n in e) n in $a || (t[n] = e[n]);
	if (e.css === !1) return t;
	let { name: n = "v", type: r, duration: i, enterFromClass: a = `${n}-enter-from`, enterActiveClass: o = `${n}-enter-active`, enterToClass: c = `${n}-enter-to`, appearFromClass: l = a, appearActiveClass: u = o, appearToClass: d = c, leaveFromClass: f = `${n}-leave-from`, leaveActiveClass: p = `${n}-leave-active`, leaveToClass: m = `${n}-leave-to` } = e, h = ao(i), g = h && h[0], _ = h && h[1], { onBeforeEnter: v, onEnter: y, onEnterCancelled: b, onLeave: x, onLeaveCancelled: S, onBeforeAppear: C = v, onAppear: w = y, onAppearCancelled: ee = b } = t, te = (e, t, n, r) => {
		e._enterCancelled = r, co(e, t ? d : c), co(e, t ? u : o), n && n();
	}, ne = (e, t) => {
		e._isLeaving = !1, co(e, f), co(e, m), co(e, p), t && t();
	}, T = (e) => (t, n) => {
		let i = e ? w : y, o = () => te(t, e, n);
		no(i, [t, o]), lo(() => {
			co(t, e ? l : a), so(t, e ? d : c), ro(i) || fo(t, r, g, o);
		});
	};
	return s(t, {
		onBeforeEnter(e) {
			no(v, [e]), so(e, a), so(e, o);
		},
		onBeforeAppear(e) {
			no(C, [e]), so(e, l), so(e, u);
		},
		onEnter: T(!1),
		onAppear: T(!0),
		onLeave(e, t) {
			e._isLeaving = !0;
			let n = () => ne(e, t);
			so(e, f), e._enterCancelled ? (so(e, p), go(e)) : (go(e), so(e, p)), lo(() => {
				e._isLeaving && (co(e, f), so(e, m), ro(x) || fo(e, r, _, n));
			}), no(x, [e, n]);
		},
		onEnterCancelled(e) {
			te(e, !1, void 0, !0), no(b, [e]);
		},
		onAppearCancelled(e) {
			te(e, !0, void 0, !0), no(ee, [e]);
		},
		onLeaveCancelled(e) {
			ne(e), no(S, [e]);
		}
	});
}
function ao(e) {
	if (e == null) return null;
	if (v(e)) return [oo(e.enter), oo(e.leave)];
	{
		let t = oo(e);
		return [t, t];
	}
}
function oo(e) {
	return le(e);
}
function so(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.add(t)), (e[Qa] || (e[Qa] = /* @__PURE__ */ new Set())).add(t);
}
function co(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.remove(t));
	let n = e[Qa];
	n && (n.delete(t), n.size || (e[Qa] = void 0));
}
function lo(e) {
	requestAnimationFrame(() => {
		requestAnimationFrame(e);
	});
}
var uo = 0;
function fo(e, t, n, r) {
	let i = e._endId = ++uo, a = () => {
		i === e._endId && r();
	};
	if (n != null) return setTimeout(a, n);
	let { type: o, timeout: s, propCount: c } = po(e, t);
	if (!o) return r();
	let l = o + "end", u = 0, d = () => {
		e.removeEventListener(l, f), a();
	}, f = (t) => {
		t.target === e && ++u >= c && d();
	};
	setTimeout(() => {
		u < c && d();
	}, s + 1), e.addEventListener(l, f);
}
function po(e, t) {
	let n = window.getComputedStyle(e), r = (e) => (n[e] || "").split(", "), i = r(`${Xa}Delay`), a = r(`${Xa}Duration`), o = mo(i, a), s = r(`${Za}Delay`), c = r(`${Za}Duration`), l = mo(s, c), u = null, d = 0, f = 0;
	t === Xa ? o > 0 && (u = Xa, d = o, f = a.length) : t === Za ? l > 0 && (u = Za, d = l, f = c.length) : (d = Math.max(o, l), u = d > 0 ? o > l ? Xa : Za : null, f = u ? u === Xa ? a.length : c.length : 0);
	let p = u === Xa && /\b(?:transform|all)(?:,|$)/.test(r(`${Xa}Property`).toString());
	return {
		type: u,
		timeout: d,
		propCount: f,
		hasTransform: p
	};
}
function mo(e, t) {
	for (; e.length < t.length;) e = e.concat(e);
	return Math.max(...t.map((t, n) => ho(t) + ho(e[n])));
}
function ho(e) {
	return e === "auto" ? 0 : Number(e.slice(0, -1).replace(",", ".")) * 1e3;
}
function go(e) {
	return (e ? e.ownerDocument : document).body.offsetHeight;
}
function _o(e, t, n) {
	let r = e[Qa];
	r && (t = (t ? [t, ...r] : [...r]).join(" ")), t == null ? e.removeAttribute("class") : n ? e.setAttribute("class", t) : e.className = t;
}
var vo = /* @__PURE__ */ Symbol("_vod"), yo = /* @__PURE__ */ Symbol("_vsh"), bo = /* @__PURE__ */ Symbol(""), xo = /(?:^|;)\s*display\s*:/;
function So(e, t, n) {
	let r = e.style, i = g(n), a = !1;
	if (n && !i) {
		if (t) if (g(t)) for (let e of t.split(";")) {
			let t = e.slice(0, e.indexOf(":")).trim();
			n[t] ?? wo(r, t, "");
		}
		else for (let e in t) n[e] ?? wo(r, e, "");
		for (let i in n) {
			i === "display" && (a = !0);
			let o = n[i];
			o == null ? wo(r, i, "") : Oo(e, i, !g(t) && t ? t[i] : void 0, o) || wo(r, i, o);
		}
	} else if (i) {
		if (t !== n) {
			let e = r[bo];
			e && (n += ";" + e), r.cssText = n, a = xo.test(n);
		}
	} else t && e.removeAttribute("style");
	vo in e && (e[vo] = a ? r.display : "", e[yo] && (r.display = "none"));
}
var Co = /\s*!important$/;
function wo(e, t, n) {
	if (d(n)) n.forEach((n) => wo(e, t, n));
	else if (n ??= "", t.startsWith("--")) e.setProperty(t, n);
	else {
		let r = Do(e, t);
		Co.test(n) ? e.setProperty(E(r), n.replace(Co, ""), "important") : e[r] = n;
	}
}
var To = [
	"Webkit",
	"Moz",
	"ms"
], Eo = {};
function Do(e, t) {
	let n = Eo[t];
	if (n) return n;
	let r = T(t);
	if (r !== "filter" && r in e) return Eo[t] = r;
	r = ie(r);
	for (let n = 0; n < To.length; n++) {
		let i = To[n] + r;
		if (i in e) return Eo[t] = i;
	}
	return t;
}
function Oo(e, t, n, r) {
	return e.tagName === "TEXTAREA" && (t === "width" || t === "height") && g(r) && n === r;
}
var ko = "http://www.w3.org/1999/xlink";
function Ao(e, t, n, r, i, a = ve(t)) {
	r && t.startsWith("xlink:") ? n == null ? e.removeAttributeNS(ko, t.slice(6, t.length)) : e.setAttributeNS(ko, t, n) : n == null || a && !ye(n) ? e.removeAttribute(t) : e.setAttribute(t, a ? "" : _(n) ? String(n) : n);
}
function jo(e, t, n, r, i) {
	if (t === "innerHTML" || t === "textContent") {
		n != null && (e[t] = t === "innerHTML" ? Wa(n) : n);
		return;
	}
	let a = e.tagName;
	if (t === "value" && a !== "PROGRESS" && !a.includes("-")) {
		let r = a === "OPTION" ? e.getAttribute("value") || "" : e.value, i = n == null ? e.type === "checkbox" ? "on" : "" : String(n);
		(r !== i || !("_value" in e)) && (e.value = i), n ?? e.removeAttribute(t), e._value = n;
		return;
	}
	let o = !1;
	if (n === "" || n == null) {
		let r = typeof e[t];
		r === "boolean" ? n = ye(n) : n == null && r === "string" ? (n = "", o = !0) : r === "number" && (n = 0, o = !0);
	}
	try {
		e[t] = n;
	} catch {}
	o && e.removeAttribute(i || t);
}
function Mo(e, t, n, r) {
	e.addEventListener(t, n, r);
}
function No(e, t, n, r) {
	e.removeEventListener(t, n, r);
}
var Po = /* @__PURE__ */ Symbol("_vei");
function Fo(e, t, n, r, i = null) {
	let a = e[Po] || (e[Po] = {}), o = a[t];
	if (r && o) o.value = r;
	else {
		let [n, s] = Ro(t);
		r ? Mo(e, n, a[t] = Ho(r, i), s) : o && (No(e, n, o, s), a[t] = void 0);
	}
}
var Io = /(Once|Passive|Capture)$/, Lo = /^on:?(?:Once|Passive|Capture)$/;
function Ro(e) {
	let t, n;
	for (; (n = e.match(Io)) && !Lo.test(e);) t ||= {}, e = e.slice(0, e.length - n[1].length), t[n[1].toLowerCase()] = !0;
	return [e[2] === ":" ? e.slice(3) : E(e.slice(2)), t];
}
var zo = 0, Bo = /* @__PURE__ */ Promise.resolve(), Vo = () => zo ||= (Bo.then(() => zo = 0), Date.now());
function Ho(e, t) {
	let n = (e) => {
		if (!e._vts) e._vts = Date.now();
		else if (e._vts <= n.attached) return;
		let r = n.value;
		if (d(r)) {
			let n = e.stopImmediatePropagation;
			e.stopImmediatePropagation = () => {
				n.call(e), e._stopped = !0;
			};
			let i = r.slice(), a = [e];
			for (let n = 0; n < i.length && !e._stopped; n++) {
				let e = i[n];
				e && on(e, t, 5, a);
			}
		} else on(r, t, 5, [e]);
	};
	return n.value = e, n.attached = Vo(), n;
}
var Uo = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && e.charCodeAt(2) > 96 && e.charCodeAt(2) < 123, Wo = (e, t, n, r, i, s) => {
	let c = i === "svg";
	t === "class" ? _o(e, r, c) : t === "style" ? So(e, n, r) : a(t) ? o(t) || Fo(e, t, n, r, s) : (t[0] === "." ? (t = t.slice(1), !0) : t[0] === "^" ? (t = t.slice(1), !1) : Go(e, t, r, c)) ? (jo(e, t, r), !e.tagName.includes("-") && (t === "value" || t === "checked" || t === "selected") && Ao(e, t, r, c, s, t !== "value")) : e._isVueCE && (Ko(e, t) || e._def.__asyncLoader && (/[A-Z]/.test(t) || !g(r))) ? jo(e, T(t), r, s, t) : (t === "true-value" ? e._trueValue = r : t === "false-value" && (e._falseValue = r), Ao(e, t, r, c));
};
function Go(e, t, n, r) {
	if (r) return !!(t === "innerHTML" || t === "textContent" || t in e && Uo(t) && h(n));
	if (t === "spellcheck" || t === "draggable" || t === "translate" || t === "autocorrect" || t === "sandbox" && e.tagName === "IFRAME" || t === "form" || t === "list" && e.tagName === "INPUT" || t === "type" && e.tagName === "TEXTAREA") return !1;
	if (t === "width" || t === "height") {
		let t = e.tagName;
		if (t === "IMG" || t === "VIDEO" || t === "CANVAS" || t === "SOURCE") return !1;
	}
	return Uo(t) && g(n) ? !1 : t in e;
}
function Ko(e, t) {
	let n = e._def.props;
	if (!n) return !1;
	let r = T(t);
	return Array.isArray(n) ? n.some((e) => T(e) === r) : Object.keys(n).some((e) => T(e) === r);
}
var qo = (e) => {
	let t = e.props["onUpdate:modelValue"] || !1;
	return d(t) ? (e) => se(t, e) : t;
};
function Jo(e) {
	e.target.composing = !0;
}
function Yo(e) {
	let t = e.target;
	t.composing && (t.composing = !1, t.dispatchEvent(new Event("input")));
}
var Xo = /* @__PURE__ */ Symbol("_assign");
function Zo(e, t, n) {
	return t && (e = e.trim()), n && (e = ce(e)), e;
}
var Qo = {
	created(e, { modifiers: { lazy: t, trim: n, number: r } }, i) {
		e[Xo] = qo(i);
		let a = r || i.props && i.props.type === "number";
		Mo(e, t ? "change" : "input", (t) => {
			t.target.composing || e[Xo](Zo(e.value, n, a));
		}), (n || a) && Mo(e, "change", () => {
			e.value = Zo(e.value, n, a);
		}), t || (Mo(e, "compositionstart", Jo), Mo(e, "compositionend", Yo), Mo(e, "change", Yo));
	},
	mounted(e, { value: t }) {
		e.value = t ?? "";
	},
	beforeUpdate(e, { value: t, oldValue: n, modifiers: { lazy: r, trim: i, number: a } }, o) {
		if (e[Xo] = qo(o), e.composing) return;
		let s = (a || e.type === "number") && !/^0\d/.test(e.value) ? ce(e.value) : e.value, c = t ?? "";
		if (s === c) return;
		let l = e.getRootNode();
		(l instanceof Document || l instanceof ShadowRoot) && l.activeElement === e && e.type !== "range" && (r && t === n || i && e.value.trim() === c) || (e.value = c);
	}
}, $o = {
	deep: !0,
	created(e, t, n) {
		e[Xo] = qo(n), Mo(e, "change", () => {
			let t = e._modelValue, n = rs(e), r = e.checked, i = e[Xo];
			if (d(t)) {
				let e = Se(t, n), a = e !== -1;
				if (r && !a) i(t.concat(n));
				else if (!r && a) {
					let n = [...t];
					n.splice(e, 1), i(n);
				}
			} else if (p(t)) {
				let e = new Set(t);
				r ? e.add(n) : e.delete(n), i(e);
			} else i(is(e, r));
		});
	},
	mounted: es,
	beforeUpdate(e, t, n) {
		e[Xo] = qo(n), es(e, t, n);
	}
};
function es(e, { value: t, oldValue: n }, r) {
	e._modelValue = t;
	let i;
	if (d(t)) i = Se(t, r.props.value) > -1;
	else if (p(t)) i = t.has(r.props.value);
	else {
		if (t === n) return;
		i = xe(t, is(e, !0));
	}
	e.checked !== i && (e.checked = i);
}
var ts = {
	deep: !0,
	created(e, { value: t, modifiers: { number: n } }, r) {
		e._modelValue = t, Mo(e, "change", () => {
			let t = Array.prototype.filter.call(e.options, (e) => e.selected).map((e) => n ? ce(rs(e)) : rs(e));
			e[Xo](e.multiple ? p(e._modelValue) ? new Set(t) : t : t[0]), e._assigning = !0, hn(() => {
				e._assigning = !1;
			});
		}), e[Xo] = qo(r);
	},
	mounted(e, { value: t }) {
		ns(e, t);
	},
	beforeUpdate(e, { value: t }, n) {
		e._modelValue = t, e[Xo] = qo(n);
	},
	updated(e, { value: t }) {
		e._assigning || ns(e, t);
	}
};
function ns(e, t) {
	let n = e.multiple, r = d(t);
	if (!(n && !r && !p(t))) {
		for (let i = 0, a = e.options.length; i < a; i++) {
			let a = e.options[i], o = rs(a);
			if (n) if (r) {
				let e = typeof o;
				a.selected = e === "string" || e === "number" ? t.some((e) => String(e) === String(o)) : Se(t, o) > -1;
			} else a.selected = t.has(o);
			else if (xe(rs(a), t)) {
				e.selectedIndex !== i && (e.selectedIndex = i);
				return;
			}
		}
		!n && e.selectedIndex !== -1 && (e.selectedIndex = -1);
	}
}
function rs(e) {
	return "_value" in e ? e._value : e.value;
}
function is(e, t) {
	let n = t ? "_trueValue" : "_falseValue";
	return n in e ? e[n] : t;
}
var as = [
	"ctrl",
	"shift",
	"alt",
	"meta"
], os = {
	stop: (e) => e.stopPropagation(),
	prevent: (e) => e.preventDefault(),
	self: (e) => e.target !== e.currentTarget,
	ctrl: (e) => !e.ctrlKey,
	shift: (e) => !e.shiftKey,
	alt: (e) => !e.altKey,
	meta: (e) => !e.metaKey,
	left: (e) => "button" in e && e.button !== 0,
	middle: (e) => "button" in e && e.button !== 1,
	right: (e) => "button" in e && e.button !== 2,
	exact: (e, t) => as.some((n) => e[`${n}Key`] && !t.includes(n))
}, ss = (e, t) => {
	if (!e) return e;
	let n = e._withMods ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n, ...r) => {
		for (let e = 0; e < t.length; e++) {
			let r = os[t[e]];
			if (r && r(n, t)) return;
		}
		return e(n, ...r);
	}));
}, cs = {
	esc: "escape",
	space: " ",
	up: "arrow-up",
	left: "arrow-left",
	right: "arrow-right",
	down: "arrow-down",
	delete: "backspace"
}, ls = (e, t) => {
	let n = e._withKeys ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n) => {
		if (!("key" in n)) return;
		let r = E(n.key);
		if (t.some((e) => e === r || cs[e] === r)) return e(n);
	}));
}, us = /* @__PURE__ */ s({ patchProp: Wo }, Ya), ds;
function fs() {
	return ds ||= Ri(us);
}
var ps = ((...e) => {
	let t = fs().createApp(...e), { mount: n } = t;
	return t.mount = (e) => {
		let r = hs(e);
		if (!r) return;
		let i = t._component;
		!h(i) && !i.render && !i.template && (i.template = r.innerHTML), r.nodeType === 1 && (r.textContent = "");
		let a = n(r, !1, ms(r));
		return r instanceof Element && (r.removeAttribute("v-cloak"), r.setAttribute("data-v-app", "")), a;
	}, t;
});
function ms(e) {
	if (e instanceof SVGElement) return "svg";
	if (typeof MathMLElement == "function" && e instanceof MathMLElement) return "mathml";
}
function hs(e) {
	return g(e) ? document.querySelector(e) : e;
}
//#endregion
//#region src/api.ts
var gs = /* @__PURE__ */ new Map();
async function _s(e, t = {}) {
	let n = await fetch(e, {
		credentials: "same-origin",
		cache: "no-store",
		...t,
		headers: {
			Accept: "application/json",
			...t.headers || {}
		}
	}), r = (n.headers.get("content-type") || "").includes("application/json") ? await n.json() : await n.text();
	if (!n.ok) {
		let e = typeof r == "object" && r ? r.error || r.detail || r.message : r;
		throw Error(String(e || `Request failed (${n.status})`));
	}
	return r;
}
function vs(e) {
	let t = gs.get(e);
	if (t) return t;
	let n = (async () => {
		let t = new AbortController(), n = window.setTimeout(() => t.abort(), e === "/api/data" ? 12e4 : 2e4);
		try {
			return await _s(e, { signal: t.signal });
		} finally {
			window.clearTimeout(n), gs.delete(e);
		}
	})();
	return gs.set(e, n), n;
}
function ys(e, t = {}) {
	return _s(e, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
function bs(e, t) {
	return _s(e, {
		method: "PUT",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
//#endregion
//#region src/liveUpdates.ts
function xs(e, t = window, n = document) {
	let r = null, i = 0, a = !0, o = !1, s = !1, c = !1, l = 0, u = "", d = 0;
	function f() {
		t.clearTimeout(i), !a && n.visibilityState !== "hidden" && (i = t.setTimeout(() => void p(), 1500));
	}
	async function p() {
		if (a || n.visibilityState === "hidden" || s) return;
		t.clearTimeout(i), o && Date.now() - d > 2e4 && (h(), e.state("reconnecting")), g();
		let r = c || Date.now() - l >= 5e3;
		if (c = !1, o && !r) {
			f();
			return;
		}
		s = !0;
		try {
			await e.synchronize({
				status: !o,
				panels: r
			}), r && (l = Date.now());
		} catch {} finally {
			s = !1, c && !a ? p() : f();
		}
	}
	function m() {
		c = !0, p();
	}
	function h() {
		r?.close(), r = null, o = !1, t.clearTimeout(i);
	}
	function g() {
		if (r || a || n.visibilityState === "hidden") return;
		if (typeof t.EventSource != "function") {
			e.state("polling");
			return;
		}
		e.state("connecting");
		let i;
		try {
			i = new t.EventSource("/api/events");
		} catch {
			e.state("polling");
			return;
		}
		r = i, i.onopen = () => {
			r === i && (o = !0, d = Date.now(), e.state("live"), m());
		}, i.addEventListener("trainer", (t) => {
			if (!(r !== i || a)) try {
				let n = JSON.parse(t.data);
				if (!n.training || typeof n.revision != "string") return;
				d = Date.now(), e.training(n.training), u !== n.revision && (u = n.revision, m());
			} catch {}
		}), i.addEventListener("heartbeat", () => {
			r === i && (d = Date.now());
		}), i.onerror = () => {
			r === i && (o = !1, e.state("reconnecting"), i.readyState === 2 && (i.close(), r = null), m());
		};
	}
	function _() {
		if (!a) {
			if (n.visibilityState === "hidden") {
				h(), e.state("paused");
				return;
			}
			g(), m();
		}
	}
	return {
		refresh: m,
		start() {
			a && (a = !1, n.addEventListener("visibilitychange", _), t.addEventListener("focus", _), t.addEventListener("online", _), _());
		},
		stop() {
			a = !0, c = !1, h(), n.removeEventListener("visibilitychange", _), t.removeEventListener("focus", _), t.removeEventListener("online", _);
		}
	};
}
//#endregion
//#region src/trainerStore.ts
var Ss = () => ({
	running: !1,
	exit_code: null,
	log_lines: [],
	progress: null
}), Cs = () => ({
	personal: [],
	negative: [],
	personal_count: 0,
	negative_count: 0
}), ws = () => ({
	items: [],
	captured_count: 0,
	personal_count: 0,
	negative_count: 0
}), Ts = () => ({
	items: [],
	total_size_bytes: 0,
	total_file_count: 0
}), Es = () => ({
	enabled: !1,
	wake_phrase: "",
	language: "en",
	english_accent: "mixed",
	stt_engine: "faster_whisper",
	minimum_transcript_chars: 2,
	delete_confirmed_wakes: !1,
	promote_close_misses: !1,
	schedule_hours: 24,
	minimum_new_negatives: 3,
	advertised_base_url: "",
	tater_url: "http://127.0.0.1:8501",
	notify_satellites: !0,
	train_microwakeword: !0,
	train_openwakeword: !1,
	train_nanowakeword_cnn: !1,
	train_nanowakeword_gru: !1,
	generated_negative_phrases: ""
}), J = /* @__PURE__ */ Ft({
	activeView: "home",
	selectedWord: "",
	wizardStep: 0,
	wizardRetrain: !1,
	trainingStages: [],
	trainingTargets: [],
	initialized: !1,
	liveState: "connecting",
	busy: /* @__PURE__ */ new Set(),
	phrase: "",
	language: "en",
	englishAccent: "mixed",
	ttsMode: "hybrid",
	trainMicroWakeWord: !0,
	trainOpenWakeWord: !1,
	trainNanoWakeWordCnn: !1,
	trainNanoWakeWordGru: !1,
	generatedNegativePhrases: "",
	languages: [{
		code: "en",
		label: "English (en)",
		engines: ["omnivoice"]
	}],
	englishAccents: [
		{
			code: "mixed",
			label: "Mixed English"
		},
		{
			code: "australian",
			label: "Australian"
		},
		{
			code: "american",
			label: "American"
		},
		{
			code: "british",
			label: "British"
		},
		{
			code: "canadian",
			label: "Canadian"
		},
		{
			code: "irish",
			label: "Irish"
		},
		{
			code: "scottish",
			label: "Scottish"
		},
		{
			code: "new_zealand",
			label: "New Zealand"
		},
		{
			code: "indian",
			label: "Indian"
		},
		{
			code: "south_african",
			label: "South African"
		}
	],
	session: {},
	samples: Cs(),
	captured: ws(),
	training: Ss(),
	auto: {},
	autoForm: Es(),
	wakeWords: [],
	wakeWordLibrary: [],
	modelHistory: [],
	managedData: Ts(),
	selectedFiles: [],
	sampleBucket: "personal",
	samplePage: {
		personal: 0,
		negative: 0
	},
	uploadProgress: 0,
	uploadLabel: "No upload in progress",
	uploadDetail: "Choose files and upload when you are ready.",
	consoleOpen: !1,
	taterLinkOpen: !1,
	trimItem: null,
	trimBucket: "personal",
	toast: {
		message: "",
		tone: "success",
		serial: 0
	}
}), Ds = !!(window.__wakeWordTrainerMac?.dataFolderPicker && window.webkit?.messageHandlers?.wakeWordTrainerStorage), Os = null, ks = Es(), As = /* @__PURE__ */ new Map(), js = /* @__PURE__ */ new Map(), Ms = 0, Ns = q(() => Number(J.samples.personal_count ?? J.session.takes_received ?? 0)), Ps = q(() => Number(J.samples.negative_count ?? J.captured.negative_count ?? 0)), Fs = q(() => J.languages.find((e) => e.code === J.language) || J.languages[0]), Is = q(() => {
	let e = Fs.value?.engines?.length ? Fs.value.engines : ["omnivoice"], t = J.ttsMode === "piper" ? e.filter((e) => e === "piper") : J.ttsMode === "hybrid" ? e : e.filter((e) => e !== "piper"), n = {
		omnivoice: "OmniVoice",
		qwen3: "Qwen3",
		moss: "MOSS",
		piper: "Piper"
	}, r = J.ttsMode === "piper" ? "Legacy" : Vs(Fs.value?.quality || "experimental");
	return `${t.map((e) => n[e] || e).join(" + ") || "Unavailable"} · ${r}`;
}), Ls = q(() => !!(J.training.running || J.training.exit_code !== null || J.training.log_lines?.length)), Rs = q(() => J.samples[J.sampleBucket] || []), zs = q(() => !!J.auto.trainer_link?.linked), Bs = q(() => {
	let e = J.auto.stt_engines;
	return Array.isArray(e) && e.length ? e : [{
		id: "faster_whisper",
		label: "Faster Whisper"
	}, {
		id: "parakeet_onnx",
		label: "Parakeet ONNX"
	}];
});
function Vs(e) {
	return String(e || "").replaceAll("_", " ").replace(/\b\w/g, (e) => e.toUpperCase());
}
function Y(e) {
	return e ? J.busy.has(e) : J.busy.size > 0;
}
function X(e, t) {
	t ? J.busy.add(e) : J.busy.delete(e);
}
function Z(e, t = "success") {
	J.toast = {
		message: String(e || ""),
		tone: t,
		serial: J.toast.serial + 1
	};
}
function Q(e, t) {
	Z(e instanceof Error ? e.message : t, "error");
}
function Hs(e) {
	J.training = e;
	let t = e.progress;
	if (t?.stage) {
		let n = J.trainingStages.at(-1);
		!n || n.stage !== t.stage ? (n && n.status === "running" && (n.status = t.status === "failed" ? "failed" : "complete"), J.trainingStages.push({
			stage: t.stage,
			label: t.label,
			status: t.status || (e.running ? "running" : e.exit_code === 0 ? "complete" : "failed")
		}), J.trainingStages.length > 40 && J.trainingStages.shift()) : (n.label = t.label, n.status = t.status || (e.running ? "running" : e.exit_code === 0 ? "complete" : "failed"));
	}
}
function Us(e, t = !0, n = !0) {
	let r = J.session;
	J.session = e || {}, Array.isArray(e.available_languages) && e.available_languages.length && (J.languages = e.available_languages), Array.isArray(e.available_english_accents) && e.available_english_accents.length && (J.englishAccents = e.available_english_accents), e.raw_phrase && (t || J.phrase === (r.raw_phrase || "")) && (J.phrase = e.raw_phrase), e.language && (t || J.language === (r.language || "en")) && (J.language = e.language), e.english_accent && (t || J.englishAccent === (r.english_accent || "mixed")) && (J.englishAccent = e.english_accent), e.tts_mode && (t || J.ttsMode === (r.tts_mode || "hybrid")) && (J.ttsMode = e.tts_mode), t && n && e.training && Hs(e.training);
}
async function Ws(e = !0) {
	let t = await vs("/api/session");
	return Us(t, e, !Os), t;
}
async function Gs() {
	if (!J.phrase.trim()) {
		Z("Enter a wake phrase first.", "warning");
		return;
	}
	X("session", !0);
	try {
		let e = await ys("/api/start_session", {
			phrase: J.phrase.trim(),
			language: J.language,
			english_accent: J.englishAccent,
			tts_mode: J.ttsMode
		});
		Us(e), await dc(!0), Z(`Session ${e.safe_word || "started"} is ready.`);
	} catch (e) {
		Q(e, "Session failed to start.");
	} finally {
		X("session", !1);
	}
}
async function Ks() {
	let e = !!J.training.running;
	if (!(e && !window.confirm("Training is running. Stop training cleanly and end this session?"))) {
		X("session", !0);
		try {
			Us(await ys("/api/stop_session")), Z(e ? "Training stopped cleanly and the session ended." : "Session ended. You can edit the wake phrase now.");
		} catch (t) {
			e && wc(), Q(t, "Session could not be stopped.");
		} finally {
			X("session", !1);
		}
	}
}
function qs() {
	if (!J.phrase.trim() || !("speechSynthesis" in window)) return;
	let e = new SpeechSynthesisUtterance(J.phrase.trim());
	e.lang = J.language, window.speechSynthesis.cancel(), window.speechSynthesis.speak(e);
}
function Js() {
	let e = Fs.value?.engines || [], t = e.some((e) => e !== "piper"), n = e.includes("piper");
	J.ttsMode === "modern" && !t && (J.ttsMode = "piper"), J.ttsMode === "hybrid" && !(t && n) && (J.ttsMode = t ? "modern" : "piper"), J.ttsMode === "piper" && !n && (J.ttsMode = "modern"), (J.language !== "en" || J.ttsMode === "piper") && (J.englishAccent = "mixed");
}
async function Ys(e = !1) {
	e || X("samples", !0);
	try {
		let e = await vs("/api/samples");
		J.samples = {
			...Cs(),
			...e
		};
		for (let e of ["personal", "negative"]) {
			let t = Math.max(0, Math.ceil((J.samples[e]?.length || 0) / 50) - 1);
			J.samplePage[e] = Math.min(J.samplePage[e], t);
		}
		return e;
	} finally {
		e || X("samples", !1);
	}
}
async function Xs(e = !1) {
	e || X("captured", !0);
	try {
		let e = await vs("/api/captured_audio"), t = new Map(J.captured.items.map((e) => [`${e.capture_scope || ""}/${e.saved_as}`, e])), n = /* @__PURE__ */ new Map();
		for (let r of e.items || []) {
			let e = `${r.capture_scope || ""}/${r.saved_as}`, i = t.get(e);
			n.set(e, r.review_wake_word), i && As.has(e) && i.review_wake_word !== As.get(e) && (r.review_wake_word = i.review_wake_word);
		}
		return As.clear(), n.forEach((e, t) => As.set(t, e)), J.captured = {
			...ws(),
			...e
		}, e;
	} finally {
		e || X("captured", !1);
	}
}
function Zs(e) {
	let t = e.target;
	J.selectedFiles = Array.from(t.files || []);
}
function Qs(e, t, n) {
	return new Promise((r, i) => {
		let a = new XMLHttpRequest(), o = new FormData();
		o.append("file", e, e.name), a.open("POST", "/api/upload_personal_sample"), a.responseType = "json", a.upload.onprogress = (r) => {
			r.lengthComputable && (J.uploadProgress = Math.round((t + r.loaded / r.total) / n * 100), J.uploadLabel = `Uploading ${e.name} (${t + 1}/${n})`, J.uploadDetail = "Sending and normalizing the recording.");
		}, a.onload = () => {
			let t = a.response || {};
			a.status >= 200 && a.status < 300 ? r(t) : i(Error(t.error || `Upload failed for ${e.name}`));
		}, a.onerror = () => i(/* @__PURE__ */ Error(`Upload failed for ${e.name}`)), a.send(o);
	});
}
async function $s(e) {
	if (!J.session.safe_word) {
		Z("Start a trainer session before uploading samples.", "warning");
		return;
	}
	if (J.selectedFiles.length) {
		X("upload", !0);
		try {
			let t = [...J.selectedFiles];
			for (let e = 0; e < t.length; e += 1) await Qs(t[e], e, t.length);
			J.uploadProgress = 100, J.uploadLabel = "Upload complete", J.uploadDetail = `${t.length} sample${t.length === 1 ? "" : "s"} saved in the required training format.`, J.selectedFiles = [], e && (e.value = ""), await Promise.all([Ws(), Ys(!0)]), Z("Personal samples uploaded.");
		} catch (e) {
			J.uploadProgress = 0, Q(e, "Sample upload failed.");
		} finally {
			X("upload", !1);
		}
	}
}
async function ec(e, t) {
	if (t === "discard" && !window.confirm(`Discard ${e.saved_as} from the captured-audio inbox?`)) return;
	let n = String(e.capture_scope || "").trim(), r = String(e.review_wake_word || e.dataset_wake_word || "").trim();
	if (t !== "discard" && !r) {
		Z("Choose the saved wake word this clip should train first.", "warning");
		return;
	}
	let i = new URLSearchParams();
	n && i.set("capture_word", n), t !== "discard" && i.set("target_word", r), X("review", !0);
	try {
		await ys(`/api/captured_audio/${encodeURIComponent(e.saved_as)}/${t}?${i.toString()}`), await Promise.all([
			Ws(),
			Xs(!0),
			Ys(!0)
		]), Z(t === "approve_personal" ? `Clip added to ${r} positives.` : t === "mark_negative" ? `Clip added to ${r} negatives.` : "Clip discarded.");
	} catch (e) {
		Q(e, "Review action failed.");
	} finally {
		X("review", !1);
	}
}
async function tc(e, t) {
	if (window.confirm(`Remove ${e.saved_as} from ${t} samples?`)) {
		X("review", !0);
		try {
			await _s(`/api/samples/${t}/${encodeURIComponent(e.saved_as)}`, { method: "DELETE" }), await Ys(!0), Z("Sample removed.");
		} catch (e) {
			Q(e, "Sample removal failed.");
		} finally {
			X("review", !1);
		}
	}
}
async function nc(e, t) {
	if (!window.confirm(`Revert ${e.saved_as} to its pre-trim version?`)) return;
	let n = new FormData();
	n.append("bucket", t), n.append("file_name", e.saved_as), X("review", !0);
	try {
		await _s("/api/samples/revert", {
			method: "POST",
			body: n
		}), await Ys(!0), Z("Original sample restored.");
	} catch (e) {
		Q(e, "Sample revert failed.");
	} finally {
		X("review", !1);
	}
}
async function rc(e) {
	let t = e === "personal" ? Ns.value : Ps.value;
	if (!(!t || !window.confirm(`Clear ${t} ${e} sample${t === 1 ? "" : "s"}?`))) {
		X("review", !0);
		try {
			await ys(e === "personal" ? "/api/reset_recordings" : "/api/reset_negative_samples"), await Promise.all([
				Ws(),
				Ys(!0),
				Xs(!0)
			]), Z(`${Vs(e)} samples cleared.`);
		} catch (e) {
			Q(e, "Samples could not be cleared.");
		} finally {
			X("review", !1);
		}
	}
}
function ic(e, t) {
	J.auto = e || {};
	let n = {
		...Es(),
		...e.config || {}
	};
	Array.isArray(e.config?.generated_negative_phrases) && (n.generated_negative_phrases = e.config.generated_negative_phrases.join("\n")), n.wake_phrase ||= J.session.raw_phrase || "", n.language ||= J.session.language || "en", n.english_accent ||= J.session.english_accent || "mixed", String(n.language).toLowerCase().startsWith("en") || (n.english_accent = "mixed");
	for (let e of Object.keys(n)) (t || J.autoForm[e] === ks[e]) && Object.assign(J.autoForm, { [e]: n[e] });
	ks = { ...n };
}
async function ac(e = !1) {
	let t = await vs("/api/auto_train");
	return ic(t, e), t;
}
async function oc() {
	if (![
		J.autoForm.train_microwakeword,
		J.autoForm.train_openwakeword,
		J.autoForm.train_nanowakeword_cnn,
		J.autoForm.train_nanowakeword_gru
	].some(Boolean)) {
		Z("Select at least one model for Auto Training.", "warning");
		return;
	}
	X("auto", !0);
	try {
		let e = await bs("/api/auto_train", J.autoForm);
		ic(e, !0), Z(e.config?.enabled ? "Auto Training saved and enabled." : "Auto Training saved.");
	} catch (e) {
		Q(e, "Auto Training settings failed to save.");
	} finally {
		X("auto", !1);
	}
}
async function sc(e) {
	X("auto", !0);
	try {
		let t = await ys("/api/auto_train/action", { action: e });
		if (ic(t, !1), e === "train_now") {
			J.trainingStages = [];
			let e = J.auto.config || J.autoForm;
			J.trainingTargets = [
				...e.train_microwakeword ? ["MicroWakeWord"] : [],
				...e.train_openwakeword ? ["openWakeWord"] : [],
				...e.train_nanowakeword_cnn ? ["NanoWakeWord CNN"] : [],
				...e.train_nanowakeword_gru ? ["NanoWakeWord GRU"] : []
			], J.training = {
				...Ss(),
				running: !0
			}, J.activeView = "training", wc();
		}
		Z(e === "review_now" ? `${Number(t.queued || 0)} clips queued for review.` : e === "train_now" ? "Training started." : "Wake word published.");
	} catch (e) {
		Q(e, "Auto Training action failed.");
	} finally {
		X("auto", !1);
	}
}
async function cc(e, t) {
	X("link", !0);
	try {
		return await ys("/api/tater_link/claim", {
			tater_url: e.trim(),
			pairing_code: t.trim()
		}), J.autoForm.tater_url = e.trim(), await ac(!1), Z("Trainer linked securely to Tater."), !0;
	} catch (e) {
		return Q(e, "Tater link failed."), !1;
	} finally {
		X("link", !1);
	}
}
async function lc() {
	if (window.confirm("Unlink this trainer from Tater?")) {
		X("auto", !0);
		try {
			await ys("/api/tater_link/unlink"), await ac(!1), Z("Trainer unlinked from Tater.", "warning");
		} catch (e) {
			Q(e, "Tater unlink failed.");
		} finally {
			X("auto", !1);
		}
	}
}
async function uc(e = !1) {
	e || X("firmware", !0);
	try {
		let e = await vs("/api/trained_wake_words/catalog");
		J.wakeWords = Array.isArray(e.wake_words) ? e.wake_words : [];
	} finally {
		e || X("firmware", !1);
	}
}
async function dc(e = !1, t = !1) {
	e || X("library", !0);
	try {
		let e = await vs(`/api/wake_word_library${t ? "?include_usage=false" : ""}`), n = new Map(J.wakeWordLibrary.map((e) => [e.safe_word, e]));
		J.wakeWordLibrary = Array.isArray(e.words) ? e.words.map((e) => ({
			...n.get(e.safe_word),
			...e
		})) : [], J.modelHistory = (await Promise.all(J.wakeWordLibrary.map(async (e) => {
			let n = JSON.stringify([
				e.run_count,
				e.published_run_id,
				e.latest_run
			]);
			if (t && js.get(e.safe_word) === n) return J.modelHistory.filter((t) => t.safe_word === e.safe_word);
			try {
				let t = await vs(`/api/wake_word_library/${encodeURIComponent(e.safe_word)}/history`);
				return js.set(e.safe_word, n), Array.isArray(t.runs) ? t.runs : [];
			} catch {
				return J.modelHistory.filter((t) => t.safe_word === e.safe_word);
			}
		}))).flat().sort((e, t) => String(t.completed_at || "").localeCompare(String(e.completed_at || "")));
	} finally {
		e || X("library", !1);
	}
}
async function fc(e) {
	if (J.session.safe_word && J.session.safe_word !== e.safe_word) {
		Z("Stop the current session before switching saved wake words.", "warning");
		return;
	}
	if (J.session.safe_word === e.safe_word) {
		Z(`${e.phrase} is already active.`);
		return;
	}
	J.phrase = e.phrase, e.language && (J.language = e.language), e.english_accent && (J.englishAccent = e.english_accent), e.tts_mode && (J.ttsMode = e.tts_mode), Js(), await Gs(), await Promise.all([Ys(!0), dc(!0)]);
}
async function pc(e, t) {
	if (J.training.running) {
		Z("Wait for training to finish before changing the published version.", "warning");
		return;
	}
	X("version-publish", !0);
	try {
		await ys(`/api/wake_word_library/${encodeURIComponent(e.safe_word)}/runs/${encodeURIComponent(t.run_id)}/publish`), await Promise.all([uc(!0), dc(!0)]), Z(`Version ${t.version || t.run_id} is now current for ${e.phrase}. Publish through Tater when you want satellites to use it.`);
	} catch (e) {
		Q(e, "The published version could not be changed.");
	} finally {
		X("version-publish", !1);
	}
}
async function mc(e = !1) {
	e || X("data", !0);
	try {
		let e = await vs("/api/data");
		return J.managedData = {
			...Ts(),
			...e
		}, Ms = Date.now(), e;
	} finally {
		e || X("data", !1);
	}
}
function hc() {
	let e = window.webkit?.messageHandlers?.wakeWordTrainerStorage;
	if (!Ds || !e) {
		Z("Open this page in the WakeWord Trainer Mac app to change the data folder.", "warning");
		return;
	}
	e.postMessage({ action: "choose" });
}
async function gc(e) {
	if (!e.file_count) return;
	let t = `${kc(e.size_bytes)} · ${Number(e.file_count).toLocaleString()} file${e.file_count === 1 ? "" : "s"}`, n = e.rebuild_note ? `\n\n${e.rebuild_note}` : "";
	if (window.confirm(`Permanently delete ${e.label} (${t})?${n}\n\nThis cannot be undone.`)) {
		X("data-delete", !0);
		try {
			let t = await _s(`/api/data/${encodeURIComponent(e.id)}`, { method: "DELETE" });
			J.managedData = {
				...Ts(),
				...t
			}, await Promise.allSettled([
				Ws(),
				Ys(!0),
				Xs(!0),
				uc(!0)
			]), Z(`${e.label} deleted. ${kc(e.size_bytes)} released.`);
		} catch (t) {
			Q(t, `${e.label} could not be deleted.`);
		} finally {
			X("data-delete", !1);
		}
	}
}
async function _c(e) {
	try {
		await navigator.clipboard.writeText(e), Z("Wake-word JSON URL copied.");
	} catch (e) {
		Q(e, "Clipboard unavailable.");
	}
}
async function vc() {
	if (![
		J.trainMicroWakeWord,
		J.trainOpenWakeWord,
		J.trainNanoWakeWordCnn,
		J.trainNanoWakeWordGru
	].some(Boolean)) {
		Z("Select at least one wake-word model to train.", "warning");
		return;
	}
	await Promise.all([Ws(), Ys(!0)]);
	let e = !1;
	if (!(!Ns.value && (e = window.confirm("No positive samples are saved. Train anyway without personal voices?"), !e))) {
		J.trainingTargets = [
			...J.trainMicroWakeWord ? ["MicroWakeWord", "Wake boundary"] : [],
			...J.trainOpenWakeWord ? ["openWakeWord"] : [],
			...J.trainNanoWakeWordCnn ? ["NanoWakeWord CNN"] : [],
			...J.trainNanoWakeWordGru ? ["NanoWakeWord GRU"] : []
		], X("training-start", !0), J.trainingStages = [], J.training = {
			running: !0,
			exit_code: null,
			log_lines: ["Waiting for training output…"]
		}, J.activeView = "training";
		try {
			await ys("/api/train", {
				allow_no_personal: e,
				train_microwakeword: J.trainMicroWakeWord,
				train_openwakeword: J.trainOpenWakeWord,
				train_nanowakeword_cnn: J.trainNanoWakeWordCnn,
				train_nanowakeword_gru: J.trainNanoWakeWordGru,
				generated_negative_phrases: J.generatedNegativePhrases
			}), wc();
		} catch (e) {
			J.training = {
				running: !1,
				exit_code: 1,
				log_lines: [e instanceof Error ? e.message : String(e)]
			}, Q(e, "Training could not start.");
		} finally {
			X("training-start", !1);
		}
	}
}
async function yc() {
	if (!J.session.safe_word) {
		Z("Select or start a saved wake word first.", "warning");
		return;
	}
	if (!J.wakeWordLibrary.find((e) => e.safe_word === J.session.safe_word)?.boundary_only_ready) {
		Z("Run one full MicroWakeWord training for this word before training its boundary only.", "warning");
		return;
	}
	J.trainingTargets = ["Wake boundary"], X("boundary-start", !0), J.trainingStages = [], J.training = {
		running: !0,
		exit_code: null,
		log_lines: ["Preparing cached data for boundary-only training…"]
	}, J.activeView = "training";
	try {
		await ys("/api/train", {
			allow_no_personal: !0,
			train_boundary_only: !0,
			train_microwakeword: !1,
			train_openwakeword: !1,
			train_nanowakeword_cnn: !1,
			train_nanowakeword_gru: !1,
			generated_negative_phrases: J.generatedNegativePhrases
		}), wc();
	} catch (e) {
		J.training = {
			running: !1,
			exit_code: 1,
			log_lines: [e instanceof Error ? e.message : String(e)]
		}, Q(e, "Boundary training could not start.");
	} finally {
		X("boundary-start", !1);
	}
}
async function bc(e = "") {
	J.trainingTargets = ["openWakeWord"], X("oww-retry", !0), J.trainingStages = [], J.training = {
		running: !0,
		exit_code: null,
		log_lines: ["Starting openWakeWord-only retry…"]
	}, J.activeView = "training";
	try {
		await ys("/api/train_openwakeword", { safe_word: e }), wc();
	} catch (e) {
		J.training = {
			running: !1,
			exit_code: 1,
			log_lines: [e instanceof Error ? e.message : String(e)]
		}, Q(e, "openWakeWord retry could not start.");
	} finally {
		X("oww-retry", !1);
	}
}
async function xc(e, t = "") {
	let n = e.toLowerCase();
	J.trainingTargets = [`NanoWakeWord ${n.toUpperCase()}`], X("nww-revalidate", !0), J.trainingStages = [], J.training = {
		running: !0,
		exit_code: null,
		log_lines: [`Revalidating existing NanoWakeWord ${n.toUpperCase()} candidate…`]
	}, J.activeView = "training";
	try {
		await ys("/api/revalidate_nanowakeword", {
			architecture: n,
			safe_word: t
		}), wc();
	} catch (e) {
		J.training = {
			running: !1,
			exit_code: 1,
			log_lines: [e instanceof Error ? e.message : String(e)]
		}, Q(e, "NanoWakeWord candidate revalidation could not start.");
	} finally {
		X("nww-revalidate", !1);
	}
}
function Sc(e) {
	if (Y("session") || [
		"training-start",
		"boundary-start",
		"oww-retry",
		"nww-revalidate"
	].some(Y)) return;
	let t = J.training.running;
	!t && e.running && (J.trainingStages = [], J.trainingTargets = []), Hs({
		...Ss(),
		...e
	}), t && !e.running && (Os?.refresh(), e.exit_code !== null && Z(e.exit_code === 0 ? "Training finished successfully." : `Training ended with exit ${e.exit_code}.`, e.exit_code === 0 ? "success" : "error"));
}
async function Cc(e) {
	if (Y()) return;
	let t = [];
	e.status && t.push(vs("/api/train_status").then((e) => Sc(e.training || Ss()))), e.panels && (t.push(Ws(!1), Ys(!0), Xs(!0), ac(!1), uc(!0), dc(!0, !0)), J.activeView === "data" && Date.now() - Ms >= 3e4 && (Ms = Date.now(), mc(!0).catch(() => void 0))), await Promise.allSettled(t);
}
function wc() {
	Os?.refresh();
}
var Tc = 0;
async function Ec() {
	let e = ++Tc;
	X("bootstrap", !0);
	try {
		await Promise.allSettled([
			Ws(),
			Ys(!0),
			Xs(!0),
			ac(!0),
			uc(!0),
			dc(!0)
		]), Js(), J.training.running && (J.activeView = "training"), J.initialized = !0;
	} finally {
		X("bootstrap", !1), e === Tc && (Os?.stop(), Os = xs({
			synchronize: Cc,
			training: (e) => Sc(e),
			state: (e) => {
				J.liveState = e;
			}
		}), Os.start());
	}
}
function Dc() {
	Tc += 1, Os?.stop(), Os = null;
}
function Oc(e) {
	if (!e) return "";
	let t = new Date(String(e));
	return Number.isNaN(t.getTime()) ? String(e) : t.toLocaleString();
}
function kc(e) {
	let t = Math.max(0, Number(e) || 0);
	if (t < 1024) return `${Math.round(t)} B`;
	let n = [
		"KB",
		"MB",
		"GB",
		"TB"
	], r = t / 1024, i = n[0];
	for (let e = 1; e < n.length && r >= 1024; e += 1) r /= 1024, i = n[e];
	return `${r >= 10 ? r.toFixed(1) : r.toFixed(2)} ${i}`;
}
function Ac(e) {
	if (!e) return "16 kHz · mono · 16-bit WAV";
	let t = Number(e.sample_rate || e.sample_rate_hz || 16e3), n = Number(e.channels || 1) === 1 ? "mono" : `${e.channels} channels`, r = Number(e.bits_per_sample || e.sample_width_bits || 16);
	return `${Math.round(t / 1e3)} kHz · ${n} · ${r}-bit`;
}
function jc(e) {
	if (e.blocked_by_vad) return {
		label: "Blocked by VAD",
		tone: "warning"
	};
	let t = String(e.event_type || "").toLowerCase();
	return t.includes("close") ? {
		label: e.capture_label || "Close miss",
		tone: "warning"
	} : t.includes("false") ? {
		label: e.capture_label || "False trigger",
		tone: "error"
	} : t.includes("wake") || t.includes("detect") ? {
		label: e.capture_label || "Wake trigger",
		tone: "success"
	} : {
		label: e.capture_label || "Captured",
		tone: "neutral"
	};
}
function Mc(e, t) {
	return e.audio_url || `/api/audio/${t}/${encodeURIComponent(e.saved_as)}`;
}
//#endregion
//#region src/components/AudioTrimModal.vue?vue&type=script&setup=true&lang.ts
var Nc = {
	class: "modal trim-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Trim audio"
}, Pc = { class: "modal-head" }, Fc = {
	key: 0,
	class: "empty-state"
}, Ic = { class: "range-grid" }, Lc = ["max"], Rc = ["min", "max"], zc = { class: "row space" }, Bc = { class: "pill" }, Vc = {
	key: 0,
	class: "pill success"
}, Hc = { class: "row modal-actions" }, Uc = ["disabled"], Wc = /* @__PURE__ */ fr({
	__name: "AudioTrimModal",
	setup(e) {
		let t = /* @__PURE__ */ F(null), n = /* @__PURE__ */ F(null), r = /* @__PURE__ */ F(0), i = /* @__PURE__ */ F(0), a = /* @__PURE__ */ F(0), o = /* @__PURE__ */ F([]), s = /* @__PURE__ */ F(!1), c = /* @__PURE__ */ F(!1);
		Nn(() => J.trimItem, async (e) => {
			if (!e) {
				n.value = null;
				return;
			}
			s.value = !0;
			try {
				let t = `/api/audio/${encodeURIComponent(J.trimBucket)}/${encodeURIComponent(e.saved_as)}`, s = await fetch(t);
				if (!s.ok) throw Error("Audio could not be loaded.");
				let c = new (window.AudioContext || window.webkitAudioContext)();
				n.value = await c.decodeAudioData(await s.arrayBuffer()), r.value = n.value.duration, i.value = 0, a.value = r.value, await c.close();
				try {
					let t = await _s(`/api/samples/${encodeURIComponent(J.trimBucket)}/${encodeURIComponent(e.saved_as)}/vad`, { method: "POST" });
					o.value = Array.isArray(t.segments) ? t.segments : [], o.value.length && (i.value = Math.max(0, Number(o.value[0].start || 0)), a.value = Math.min(r.value, Number(o.value[0].end || r.value)));
				} catch {
					o.value = [];
				}
				await hn(), d();
			} catch (e) {
				Z(e instanceof Error ? e.message : "Audio could not be loaded.", "error"), l();
			} finally {
				s.value = !1;
			}
		}, { immediate: !0 }), Nn([i, a], () => d());
		function l() {
			J.trimItem = null, n.value = null, o.value = [];
		}
		function u() {
			let e = o.value[0];
			e && (i.value = Number(e.start), a.value = Number(e.end));
		}
		function d() {
			let e = t.value, s = n.value;
			if (!e || !s || !r.value) return;
			let c = e.getBoundingClientRect();
			if (!c.width || !c.height) return;
			let l = window.devicePixelRatio || 1;
			e.width = Math.round(c.width * l), e.height = Math.round(c.height * l);
			let u = e.getContext("2d");
			if (!u) return;
			u.scale(l, l);
			let d = c.width, f = c.height, p = f / 2, m = s.getChannelData(0), h = Math.max(1, Math.floor(m.length / d));
			u.clearRect(0, 0, d, f), u.strokeStyle = "rgba(222, 218, 212, .24)", u.lineWidth = 1, u.beginPath();
			for (let e = 0; e < d; e += 1) {
				let t = 1, n = -1;
				for (let r = 0; r < h; r += 1) {
					let i = m[Math.floor(e) * h + r] || 0;
					t = Math.min(t, i), n = Math.max(n, i);
				}
				u.moveTo(e, p + t * p * .84), u.lineTo(e, p + n * p * .84);
			}
			u.stroke();
			let g = i.value / r.value * d, _ = a.value / r.value * d;
			u.fillStyle = "rgba(8, 8, 9, .66)", u.fillRect(0, 0, g, f), u.fillRect(_, 0, d - _, f), u.fillStyle = "rgba(255, 145, 52, .12)", u.fillRect(g, 0, _ - g, f), u.strokeStyle = "#ff9134", u.lineWidth = 2;
			for (let e of [g, _]) u.beginPath(), u.moveTo(e, 0), u.lineTo(e, f), u.stroke();
			u.strokeStyle = "rgba(68, 225, 165, .55)";
			for (let e of o.value) {
				let t = e.start / r.value * d;
				u.beginPath(), u.moveTo(t, 0), u.lineTo(t, f), u.stroke();
			}
		}
		function f() {
			let e = n.value;
			if (!e) return;
			let t = new (window.AudioContext || window.webkitAudioContext)(), r = t.createBufferSource();
			r.buffer = e, r.connect(t.destination), r.start(0, i.value, Math.max(.01, a.value - i.value)), r.onended = () => void t.close();
		}
		async function p() {
			let e = n.value;
			if (!e) throw Error("Audio is not loaded.");
			let t = Math.floor(i.value * e.sampleRate), r = Math.min(Math.floor(a.value * e.sampleRate), e.length), o = 16e3, s;
			if (e.sampleRate === o) s = e.getChannelData(0).slice(t, r);
			else {
				let n = Math.max(1, Math.floor((r - t) * o / e.sampleRate)), c = new OfflineAudioContext(1, n, o), l = c.createBufferSource();
				l.buffer = e, l.connect(c.destination), l.start(0, i.value, a.value - i.value), s = (await c.startRendering()).getChannelData(0);
			}
			let c = /* @__PURE__ */ new ArrayBuffer(44 + s.length * 2), l = new DataView(c);
			l.setUint32(0, 1380533830, !1), l.setUint32(4, 36 + s.length * 2, !0), l.setUint32(8, 1463899717, !1), l.setUint32(12, 1718449184, !1), l.setUint32(16, 16, !0), l.setUint16(20, 1, !0), l.setUint16(22, 1, !0), l.setUint32(24, o, !0), l.setUint32(28, o * 2, !0), l.setUint16(32, 2, !0), l.setUint16(34, 16, !0), l.setUint32(36, 1684108385, !1), l.setUint32(40, s.length * 2, !0);
			for (let e = 0; e < s.length; e += 1) l.setInt16(44 + e * 2, Math.max(-32768, Math.min(32767, Math.round(s[e] * 32767))), !0);
			return new Blob([c], { type: "audio/wav" });
		}
		async function m() {
			let e = J.trimItem;
			if (e) {
				c.value = !0;
				try {
					let t = new FormData();
					t.append("file", await p(), "trimmed.wav"), t.append("bucket", J.trimBucket), t.append("source_file", e.saved_as), t.append("start_time", i.value.toFixed(3)), t.append("end_time", a.value.toFixed(3));
					let n = await _s("/api/samples/trim", {
						method: "POST",
						body: t
					});
					l(), await Ys(!0), Z(n.message || "Trimmed sample saved.");
				} catch (e) {
					Z(e instanceof Error ? e.message : "Trim failed.", "error");
				} finally {
					c.value = !1;
				}
			}
		}
		function h() {
			J.trimItem && d();
		}
		return window.addEventListener("resize", h), Ar(() => window.removeEventListener("resize", h)), (e, n) => (V(), aa(Jn, { to: "body" }, [I(J).trimItem ? (V(), H("div", {
			key: 0,
			class: "modal-backdrop",
			onClick: ss(l, ["self"])
		}, [U("section", Nc, [
			U("header", Pc, [U("div", null, [n[2] ||= U("span", { class: "eyebrow" }, "Audio editor", -1), U("h2", null, "Trim " + k(I(J).trimItem.saved_as), 1)]), U("button", {
				type: "button",
				class: "button ghost",
				onClick: l
			}, "Close")]),
			n[3] ||= U("p", { class: "muted" }, "Keep the spoken wake phrase and remove excess silence or noise. VAD markers appear in green.", -1),
			s.value ? (V(), H("div", Fc, "Loading waveform…")) : (V(), H(B, { key: 1 }, [
				U("canvas", {
					ref_key: "canvas",
					ref: t,
					class: "waveform"
				}, null, 512),
				U("div", Ic, [U("label", null, [U("span", null, "Start · " + k(i.value.toFixed(2)) + "s", 1), R(U("input", {
					"onUpdate:modelValue": n[0] ||= (e) => i.value = e,
					type: "range",
					min: "0",
					max: Math.max(0, a.value - .01),
					step: ".01"
				}, null, 8, Lc), [[
					Qo,
					i.value,
					void 0,
					{ number: !0 }
				]])]), U("label", null, [U("span", null, "End · " + k(a.value.toFixed(2)) + "s", 1), R(U("input", {
					"onUpdate:modelValue": n[1] ||= (e) => a.value = e,
					type: "range",
					min: Math.min(r.value, i.value + .01),
					max: r.value,
					step: ".01"
				}, null, 8, Rc), [[
					Qo,
					a.value,
					void 0,
					{ number: !0 }
				]])])]),
				U("div", zc, [U("span", Bc, "Selection " + k(Math.max(0, a.value - i.value).toFixed(2)) + "s", 1), o.value.length ? (V(), H("span", Vc, k(o.value.length) + " speech segment" + k(o.value.length === 1 ? "" : "s"), 1)) : K("", !0)]),
				U("div", Hc, [
					U("button", {
						type: "button",
						onClick: f
					}, "Play selection"),
					o.value.length ? (V(), H("button", {
						key: 0,
						type: "button",
						onClick: u
					}, "Select first VAD")) : K("", !0),
					U("button", {
						type: "button",
						class: "button primary",
						disabled: c.value,
						onClick: m
					}, k(c.value ? "Saving…" : "Save trim"), 9, Uc)
				])
			], 64))
		])])) : K("", !0)]));
	}
}), Gc = {
	class: "ui-icon",
	viewBox: "0 0 24 24",
	width: "17",
	height: "17",
	fill: "none",
	stroke: "currentColor",
	"stroke-width": "1.6",
	"stroke-linecap": "round",
	"stroke-linejoin": "round",
	"aria-hidden": "true"
}, Kc = {
	key: 0,
	d: "m3 10 9-7 9 7v10H3V10Zm6 10v-7h6v7"
}, qc = {
	key: 1,
	d: "M4 9v6m4-10v14m4-17v20m4-17v14m4-10v6"
}, Jc = {
	key: 2,
	d: "M12 5v14M5 12h14"
}, Yc = {
	key: 3,
	d: "M20 12H4m7-7-7 7 7 7"
}, Xc = {
	key: 4,
	d: "M4 12h16m-7-7 7 7-7 7"
}, Zc = {
	key: 5,
	d: "M20 7v5h-5M4 17v-5h5m-4-5a8 8 0 0 1 13-3l2 3M4 17l2 3a8 8 0 0 0 13-3"
}, Qc = {
	key: 6,
	d: "M2 12h4l3-9 6 18 3-9h4"
}, $c = {
	key: 7,
	d: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"
}, el = {
	key: 8,
	d: "M14 3h7v7m0-7-11 11M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4"
}, tl = {
	key: 9,
	d: "M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"
}, nl = {
	key: 12,
	d: "m3 4-2 12v5h22v-5L21 4H3Zm-2 12h7l2 3h4l2-3h7"
}, rl = {
	key: 13,
	d: "M3 3v6h6M3 9a9 9 0 1 1-1 7M12 7v5l4 2"
}, il = {
	key: 15,
	d: "m10 13 4-4m-6 6-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m2 2 2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0"
}, al = {
	key: 16,
	x: "4",
	y: "4",
	width: "16",
	height: "16",
	rx: "2"
}, ol = {
	key: 17,
	cx: "12",
	cy: "12",
	r: "8"
}, $ = /* @__PURE__ */ fr({
	__name: "UiIcon",
	props: { name: {} },
	setup(e) {
		return (t, n) => (V(), H("svg", Gc, [e.name === "home" ? (V(), H("path", Kc)) : e.name === "wave" ? (V(), H("path", qc)) : e.name === "plus" ? (V(), H("path", Jc)) : e.name === "back" ? (V(), H("path", Yc)) : e.name === "next" ? (V(), H("path", Xc)) : e.name === "refresh" ? (V(), H("path", Zc)) : e.name === "activity" ? (V(), H("path", Qc)) : e.name === "download" ? (V(), H("path", $c)) : e.name === "external" ? (V(), H("path", el)) : e.name === "folder" ? (V(), H("path", tl)) : e.name === "file" ? (V(), H(B, { key: 10 }, [n[0] ||= U("path", { d: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z" }, null, -1), n[1] ||= U("path", { d: "M14 2v6h6M8 13h8M8 17h5" }, null, -1)], 64)) : e.name === "copy" ? (V(), H(B, { key: 11 }, [n[2] ||= U("rect", {
			x: "8",
			y: "8",
			width: "13",
			height: "13",
			rx: "2"
		}, null, -1), n[3] ||= U("path", { d: "M16 8V3H3v13h5" }, null, -1)], 64)) : e.name === "inbox" ? (V(), H("path", nl)) : e.name === "history" ? (V(), H("path", rl)) : e.name === "check" ? (V(), H(B, { key: 14 }, [n[4] ||= U("circle", {
			cx: "12",
			cy: "12",
			r: "9"
		}, null, -1), n[5] ||= U("path", { d: "m8 12 3 3 5-6" }, null, -1)], 64)) : e.name === "link" ? (V(), H("path", il)) : e.name === "stop" ? (V(), H("rect", al)) : (V(), H("circle", ol))]));
	}
}), sl = { class: "welcome-workspace" }, cl = {
	key: 0,
	class: "welcome-intro"
}, ll = {
	key: 1,
	class: "workspace-heading"
}, ul = ["disabled"], dl = {
	key: 2,
	class: "welcome-tasks"
}, fl = { class: "welcome-task" }, pl = ["disabled"], ml = { class: "welcome-task" }, hl = { class: "workspace-section-title" }, gl = {
	key: 3,
	class: "panel empty-state"
}, _l = { class: "workspace-word-icon" }, vl = { class: "row" }, yl = { class: "workspace-word-actions" }, bl = ["onClick"], xl = { key: 0 }, Sl = {
	key: 4,
	class: "workspace-note"
}, Cl = /* @__PURE__ */ fr({
	__name: "TrainerWelcome",
	props: { library: { type: Boolean } },
	emits: ["create", "open"],
	setup(e) {
		let t = q(() => J.wakeWordLibrary);
		function n(e) {
			return J.modelHistory.find((t) => t.safe_word === e.safe_word && t.is_current)?.summary || (e.published_run_id ? void 0 : e.latest_run?.summary);
		}
		function r(e) {
			return (n(e)?.models.find((e) => e.id === "micro"))?.metrics.filter((e) => ["Recall", "False accepts"].includes(e.label)).map((e) => `${e.label} ${e.value}`).join(" · ") || "";
		}
		return (i, a) => (V(), H("div", sl, [
			e.library ? (V(), H("header", ll, [a[5] ||= U("div", null, [
				U("span", { class: "eyebrow" }, "Your library"),
				U("h2", null, "Wake words"),
				U("p", null, "Each phrase keeps its results, JSON links, recordings, and saved versions.")
			], -1), U("button", {
				type: "button",
				class: "primary",
				disabled: I(J).training.running,
				onClick: a[0] ||= (e) => i.$emit("create")
			}, [W($, { name: "plus" }), a[4] ||= G("New wake word", -1)], 8, ul)])) : (V(), H("header", cl, [...a[3] ||= [
				U("span", { class: "eyebrow" }, "Wake Word Trainer", -1),
				U("h2", null, "What would you like to do?", -1),
				U("p", null, "Train a new phrase, or find the right version of one you already use.", -1)
			]])),
			e.library ? K("", !0) : (V(), H("div", dl, [U("section", fl, [
				W($, { name: "plus" }),
				a[7] ||= U("h3", null, "Train a new word", -1),
				a[8] ||= U("p", null, "Choose a phrase, add recordings, and create your first trained version.", -1),
				U("button", {
					type: "button",
					class: "primary",
					disabled: I(J).training.running,
					onClick: a[1] ||= (e) => i.$emit("create")
				}, [W($, { name: "next" }), a[6] ||= G("Create a wake word", -1)], 8, pl)
			]), U("section", ml, [
				W($, { name: "folder" }),
				a[10] ||= U("h3", null, "Use an existing word", -1),
				a[11] ||= U("p", null, "Review its results, choose a version, and get its bundle or ESPHome JSON.", -1),
				U("button", {
					type: "button",
					onClick: a[2] ||= (e) => I(J).activeView = "firmware"
				}, [W($, { name: "next" }), a[9] ||= G("Browse wake words", -1)])
			])])),
			U("div", hl, [a[12] ||= U("h3", null, "Your wake words", -1), U("small", null, k(t.value.length) + " saved " + k(t.value.length === 1 ? "word" : "words"), 1)]),
			t.value.length ? K("", !0) : (V(), H("div", gl, "Your first wake word will appear here with its recordings, models, and training history.")),
			(V(!0), H(B, null, z(t.value, (e) => (V(), H("article", {
				key: e.safe_word,
				class: "workspace-word-row"
			}, [
				U("div", _l, [W($, { name: e.has_published_bundle ? "wave" : "file" }, null, 8, ["name"])]),
				U("div", null, [U("div", vl, [U("h3", null, k(e.phrase), 1), U("span", { class: O(["pill", n(e)?.tone || ""]) }, k(n(e)?.overall || (e.has_published_bundle ? "Published · results unavailable" : "Draft")), 3)]), U("p", null, [G(k(e.run_count) + " saved " + k(e.run_count === 1 ? "run" : "runs") + " · " + k(e.personal_count || 0) + " recordings", 1), e.dataset_size_bytes == null ? K("", !0) : (V(), H(B, { key: 0 }, [G(" · " + k(I(kc)(e.dataset_size_bytes)) + " retained", 1)], 64))])]),
				U("div", yl, [U("button", {
					type: "button",
					onClick: (t) => i.$emit("open", e)
				}, [W($, { name: "external" }), a[13] ||= G("Open word", -1)], 8, bl), r(e) ? (V(), H("small", xl, "MicroWakeWord · " + k(r(e)), 1)) : K("", !0)])
			]))), 128)),
			I(J).training.running ? (V(), H("p", Sl, "A training run is in progress. You can open words and use their existing JSON links.")) : K("", !0)
		]));
	}
}), wl = { class: "workspace-heading" }, Tl = { class: "row" }, El = {
	key: 0,
	class: "pill"
}, Dl = ["disabled"], Ol = { class: "version-toolbar" }, kl = { class: "row" }, Al = {
	key: 0,
	class: "row"
}, jl = {
	key: 0,
	value: ""
}, Ml = ["value"], Nl = { key: 1 }, Pl = { key: 0 }, Fl = { key: 1 }, Il = { class: "row" }, Ll = {
	key: 0,
	class: "pill success"
}, Rl = ["disabled"], zl = {
	class: "workspace-subnav",
	role: "tablist",
	"aria-label": "Wake word workspace"
}, Bl = [
	"id",
	"aria-selected",
	"onClick"
], Vl = ["aria-labelledby"], Hl = { class: "json-link-grid" }, Ul = { class: "row" }, Wl = ["onClick"], Gl = ["href"], Kl = {
	key: 1,
	class: "workspace-note"
}, ql = { class: "panel results-panel" }, Jl = { class: "workspace-section-title" }, Yl = { class: "result-metrics" }, Xl = { class: "row space" }, Zl = { class: "workspace-advanced" }, Ql = { class: "stack" }, $l = ["href"], eu = {
	key: 0,
	class: "workspace-note"
}, tu = {
	key: 1,
	class: "workspace-note"
}, nu = { class: "panel" }, ru = { class: "workspace-section-title" }, iu = { class: "pill" }, au = {
	key: 0,
	class: "empty-state"
}, ou = {
	key: 0,
	class: "pill success"
}, su = {
	key: 1,
	class: "pill"
}, cu = { key: 0 }, lu = { class: "row" }, uu = ["onClick"], du = ["disabled", "onClick"], fu = {
	key: 0,
	class: "panel"
}, pu = { class: "workspace-section-title" }, mu = { class: "comparison-grid" }, hu = { class: "row" }, gu = { class: "pill" }, _u = ["onClick"], vu = {
	key: 2,
	class: "panel"
}, yu = { class: "workspace-section-title" }, bu = ["disabled"], xu = /* @__PURE__ */ fr({
	__name: "WordWorkspace",
	props: { word: {} },
	emits: ["retrain", "recordings"],
	setup(e) {
		let t = e, n = /* @__PURE__ */ F("results"), r = /* @__PURE__ */ F(""), i = /* @__PURE__ */ F(""), a = q(() => J.modelHistory.filter((e) => e.safe_word === t.word.safe_word)), o = q(() => a.value.find((e) => e.is_current)), s = q(() => a.value.find((e) => e.run_id === r.value)), c = q(() => J.wakeWords.find((e) => e.key === t.word.safe_word)), l = q(() => !s.value || s.value.is_current), u = q(() => s.value?.summary), d = q(() => s.value ? {
			bundle: s.value.json_url || "",
			esphome: s.value.esphome_json_url || ""
		} : {
			bundle: String(c.value?.json_url || c.value?.url || c.value?.jsonUrl || ""),
			esphome: String(c.value?.esphome_json_url || c.value?.esphomeJsonUrl || "")
		}), f = q(() => a.value.find((e) => e.run_id === i.value)), p = q(() => Array.from(new Set([...o.value?.summary?.models || [], ...f.value?.summary?.models || []].map((e) => e.id))));
		Nn(() => t.word.safe_word, () => {
			r.value = "", n.value = "results", i.value = "";
		}, { immediate: !0 }), Nn(a, () => {
			(!r.value || !a.value.some((e) => e.run_id === r.value)) && (r.value = o.value?.run_id || (t.word.has_published_bundle ? "" : a.value[0]?.run_id || ""));
		}, { immediate: !0 });
		function m(e) {
			return e ? `v${e.version || a.value.length - a.value.indexOf(e)}` : "Current publication";
		}
		function h(e) {
			r.value = e.run_id, n.value = "results", i.value = "";
		}
		function g() {
			if (s.value) return s.value.artifact_urls || {};
			let e = c.value;
			if (!e) return {};
			let t = [
				["MicroWakeWord", String(e.model_url || "")],
				["openWakeWord", String(e.openwakeword_model_url || "")],
				["Wake boundary", String(e.wake_boundary_model_url || "")]
			];
			for (let [n, r] of Object.entries(e.nanowakeword_models || {})) t.push([`NanoWakeWord ${n.toUpperCase()}`, r]);
			return Object.fromEntries(t.filter(([, e]) => e));
		}
		return (t, _) => (V(), H(B, null, [
			U("button", {
				type: "button",
				class: "workspace-back",
				onClick: _[0] ||= (e) => I(J).activeView = "home"
			}, [W($, { name: "back" }), _[7] ||= G("Back to start", -1)]),
			U("header", wl, [U("div", null, [_[8] ||= U("span", { class: "eyebrow" }, "Saved wake word", -1), U("div", Tl, [U("h2", null, k(e.word.phrase), 1), e.word.language ? (V(), H("span", El, k(e.word.language), 1)) : K("", !0)])]), U("button", {
				type: "button",
				disabled: I(J).training.running || I(Y)("session"),
				onClick: _[1] ||= (n) => t.$emit("retrain", e.word)
			}, [W($, { name: "refresh" }), _[9] ||= G("Train a new version", -1)], 8, Dl)]),
			U("div", Ol, [U("div", null, [U("div", kl, [a.value.length > 1 ? (V(), H("label", Al, [_[10] ||= G("Viewing version", -1), R(U("select", {
				"onUpdate:modelValue": _[2] ||= (e) => r.value = e,
				"aria-label": "Viewing version"
			}, [!o.value && !r.value ? (V(), H("option", jl, "Current publication")) : K("", !0), (V(!0), H(B, null, z(a.value, (e) => (V(), H("option", {
				key: e.run_id,
				value: e.run_id
			}, k(m(e)) + k(e.is_current ? " · Current" : "") + k(e === a.value[0] ? " · Latest run" : ""), 9, Ml))), 128))], 512), [[ts, r.value]])])) : (V(), H("strong", Nl, k(m(s.value)), 1)), u.value ? (V(), H("span", {
				key: 2,
				class: O(["pill", u.value.tone])
			}, k(u.value.overall), 3)) : K("", !0)]), s.value ? (V(), H("small", Pl, "Trained " + k(I(Oc)(s.value.completed_at)) + " · " + k(s.value.run_kind === "boundary" ? "Boundary only" : "Model training"), 1)) : (V(), H("small", Fl, "Published JSON links · no matching saved quality report"))]), U("div", Il, [l.value ? (V(), H("span", Ll, "Current publication")) : (V(), H(B, { key: 1 }, [U("button", {
				type: "button",
				class: "primary",
				disabled: !s.value?.restorable || I(J).training.running || I(Y)("version-publish"),
				onClick: _[3] ||= (t) => s.value && I(pc)(e.word, s.value)
			}, [W($, { name: "check" }), G(k(I(Y)("version-publish") ? "Publishing…" : `Use ${m(s.value)} as current`), 1)], 8, Rl), U("small", null, k(m(o.value)) + " is current", 1)], 64))])]),
			U("div", zl, [(V(!0), H(B, null, z([
				{
					id: "results",
					label: "Results & JSON"
				},
				{
					id: "versions",
					label: `Versions · ${a.value.length}`
				},
				{
					id: "recordings",
					label: "Recordings"
				}
			], (e) => (V(), H("button", {
				id: `word-tab-${e.id}`,
				key: e.id,
				type: "button",
				role: "tab",
				"aria-controls": "word-panel",
				"aria-selected": n.value === e.id,
				onClick: (t) => n.value = e.id
			}, k(e.label), 9, Bl))), 128))]),
			U("section", {
				id: "word-panel",
				role: "tabpanel",
				"aria-labelledby": `word-tab-${n.value}`,
				class: "stack"
			}, [n.value === "results" ? (V(), H(B, { key: 0 }, [
				U("div", Hl, [(V(!0), H(B, null, z([{
					key: "bundle",
					label: "Tater bundle JSON",
					description: "Use in Tater or the Home Assistant bridge.",
					url: d.value.bundle
				}, {
					key: "esphome",
					label: "ESPHome JSON",
					description: "Use in your ESPHome micro_wake_word setup.",
					url: d.value.esphome
				}], (e) => (V(), H("section", {
					key: e.key,
					class: "panel json-link-card"
				}, [
					U("h3", null, [W($, { name: "file" }), G(k(e.label), 1)]),
					U("p", null, k(e.description), 1),
					e.url ? (V(), H(B, { key: 0 }, [
						U("code", null, k(e.url), 1),
						U("div", Ul, [U("button", {
							type: "button",
							class: "primary",
							onClick: (t) => I(_c)(e.url)
						}, [W($, { name: "copy" }), G("Copy " + k(e.key === "bundle" ? "bundle JSON URL" : "ESPHome URL"), 1)], 8, Wl), U("a", {
							class: "button",
							href: e.url,
							target: "_blank",
							rel: "noreferrer"
						}, [W($, { name: "external" }), _[11] ||= G("View JSON", -1)], 8, Gl)]),
						U("small", null, k(s.value ? `${m(s.value)} · version-specific link` : "Current published link"), 1)
					], 64)) : (V(), H("p", Kl, k(e.key === "esphome" ? "No MicroWakeWord model was published in this version." : "No model was published by this run yet."), 1))
				]))), 128))]),
				U("section", ql, [
					U("header", Jl, [U("div", null, [U("h3", null, "Training results" + k(s.value ? ` · ${m(s.value)}` : ""), 1), U("p", null, k(u.value?.message || "Quality results are unavailable for this publication. No quality rating has been inferred."), 1)]), u.value ? (V(), H("span", {
						key: 0,
						class: O(["pill", u.value.tone])
					}, k(u.value.overall), 3)) : K("", !0)]),
					(V(!0), H(B, null, z(u.value?.models || [], (e) => (V(), H("article", {
						key: e.id,
						class: "result-model"
					}, [U("header", null, [U("div", null, [U("h3", null, k(e.label), 1), U("p", null, k(e.message), 1)]), U("span", { class: O(["pill", e.tone]) }, k(e.rating), 3)]), U("dl", Yl, [(V(!0), H(B, null, z(e.metrics, (e) => (V(), H("div", { key: e.label }, [U("dt", null, k(e.label), 1), U("dd", null, k(e.value), 1)]))), 128))])]))), 128)),
					_[12] ||= U("p", { class: "workspace-note" }, "Each detector reports its own validation metrics. Missing measurements stay unavailable.", -1)
				]),
				U("div", Xl, [U("small", null, k(a.value.length) + " saved runs · earlier results and JSON links are retained", 1), U("button", {
					type: "button",
					onClick: _[4] ||= (e) => n.value = "versions"
				}, [W($, { name: "history" }), _[13] ||= G("View version history", -1)])]),
				U("details", Zl, [_[14] ||= U("summary", null, "Raw model files & technical details", -1), U("div", Ql, [
					(V(!0), H(B, null, z(g(), (e, t) => (V(), H("div", {
						key: t,
						class: "row space"
					}, [U("span", null, k(t), 1), U("a", {
						href: e,
						target: "_blank",
						rel: "noreferrer"
					}, k(String(t).endsWith(".json") ? "View artifact" : "Download model"), 9, $l)]))), 128)),
					!s.value && c.value?.wake_boundary_available ? (V(), H("p", eu, "Wake boundary model · wake boundary · threshold " + k(c.value.wake_boundary_threshold), 1)) : K("", !0),
					s.value && !s.value.restorable ? (V(), H("p", tu, "This run did not publish a usable model. Its recorded results remain in history.")) : K("", !0)
				])])
			], 64)) : n.value === "versions" ? (V(), H(B, { key: 1 }, [U("section", nu, [
				U("header", ru, [_[15] ||= U("div", null, [U("h3", null, "Saved versions"), U("p", null, "Open any run for its exact results and immutable JSON links.")], -1), U("span", iu, k(a.value.length) + " retained", 1)]),
				a.value.length ? K("", !0) : (V(), H("div", au, "Completed training attempts will appear here. Your current JSON links are still available under Results & JSON.")),
				(V(!0), H(B, null, z(a.value, (e) => (V(), H("article", {
					key: e.run_id,
					class: "workspace-version-row"
				}, [
					U("div", null, [U("strong", null, k(m(e)), 1), e.is_current ? (V(), H("span", ou, "Current")) : e === a.value[0] ? (V(), H("span", su, "Latest run")) : K("", !0)]),
					U("div", null, [U("strong", null, k(I(Oc)(e.completed_at)), 1), U("small", null, k(e.run_kind === "boundary" ? "Boundary only" : e.selected_models?.join(" + ")), 1)]),
					U("div", null, [U("span", { class: O(["pill", e.summary?.tone || ""]) }, k(e.summary?.overall || "Results unavailable"), 3), e.restorable ? K("", !0) : (V(), H("small", cu, "No usable published model"))]),
					U("div", lu, [U("button", {
						type: "button",
						onClick: (t) => h(e)
					}, "View results", 8, uu), U("button", {
						type: "button",
						disabled: !o.value || o.value.run_id === e.run_id,
						onClick: (t) => i.value = e.run_id
					}, "Compare", 8, du)])
				]))), 128))
			]), f.value ? (V(), H("section", fu, [U("header", pu, [_[16] ||= U("div", null, [U("h3", null, "Compare with current"), U("p", null, "Training datasets can change between runs. Compare matching metrics with that context.")], -1), U("button", {
				type: "button",
				onClick: _[5] ||= (e) => i.value = ""
			}, "Close comparison")]), U("div", mu, [(V(!0), H(B, null, z([o.value, f.value].filter(Boolean), (e) => (V(), H("article", { key: e.run_id }, [
				U("header", hu, [U("h3", null, k(m(e)), 1), U("span", gu, k(e?.is_current ? "Current" : "Comparing"), 1)]),
				(V(!0), H(B, null, z(p.value, (t) => (V(), H("div", {
					key: t,
					class: "comparison-model"
				}, [
					U("h3", null, k(e?.summary?.models.find((e) => e.id === t)?.label || t), 1),
					U("p", null, k(e?.summary?.models.find((e) => e.id === t)?.rating || "Not evaluated"), 1),
					U("dl", null, [(V(!0), H(B, null, z(e?.summary?.models.find((e) => e.id === t)?.metrics || [], (e) => (V(), H("div", { key: e.label }, [U("dt", null, k(e.label), 1), U("dd", null, k(e.value), 1)]))), 128))])
				]))), 128)),
				e && !e.is_current ? (V(), H("button", {
					key: 0,
					type: "button",
					onClick: (t) => h(e)
				}, "View " + k(m(e)), 9, _u)) : K("", !0)
			]))), 128))])])) : K("", !0)], 64)) : (V(), H("section", vu, [U("header", yu, [U("div", null, [U("h3", null, "Recordings for “" + k(e.word.phrase) + "”", 1), U("p", null, k(e.word.personal_count || 0) + " personal recordings · " + k(e.word.negative_count || 0) + " reviewed negatives", 1)]), U("button", {
				type: "button",
				disabled: I(J).training.running && I(J).session.safe_word !== e.word.safe_word,
				onClick: _[6] ||= (n) => t.$emit("recordings", e.word)
			}, [W($, { name: "plus" }), _[17] ||= G("Manage recordings", -1)], 8, bu)]), _[18] ||= U("p", { class: "workspace-note" }, "Review, upload, trim, or restore recordings in the sample manager. Existing versions keep their original models and results.", -1)]))], 8, Vl)
		], 64));
	}
}), Su = { class: "app-shell trainer-shell" }, Cu = { class: "workspace-topbar" }, wu = { class: "workspace-brand" }, Tu = { class: "local-trainer-status" }, Eu = ["title"], Du = {
	class: "workspace-nav",
	"aria-label": "Trainer areas"
}, Ou = ["aria-current", "onClick"], ku = {
	key: 0,
	class: "nav-count"
}, Au = ["aria-current"], ju = { class: "workspace-more" }, Mu = {
	key: 0,
	class: "training-status-strip"
}, Nu = { class: "row" }, Pu = { class: "main-content workspace-main" }, Fu = {
	key: 0,
	class: "loading-panel",
	role: "status"
}, Iu = { class: "workspace-heading" }, Lu = { class: "eyebrow" }, Ru = {
	key: 0,
	class: "pill"
}, zu = {
	class: "wizard-steps",
	"aria-label": "Setup progress"
}, Bu = [
	"aria-current",
	"disabled",
	"onClick"
], Vu = { class: "workspace-heading" }, Hu = { class: "training-workspace-grid" }, Uu = { class: "panel training-stages-panel" }, Wu = { class: "workspace-section-title" }, Gu = { class: "console-progress-head" }, Ku = ["aria-valuenow"], qu = {
	key: 0,
	class: "console-progress-facts"
}, Ju = { class: "training-stage-list" }, Yu = {
	key: 0,
	class: "workspace-note"
}, Xu = { class: "stack" }, Zu = { class: "panel" }, Qu = { class: "training-run-summary" }, $u = { key: 0 }, ed = { class: "panel inline-training-log" }, td = { class: "workspace-section-title" }, nd = {
	key: 0,
	class: "training-summary-card"
}, rd = { class: "training-summary-grid" }, id = {
	key: 0,
	class: "summary-metrics"
}, ad = { class: "wizard-actions" }, od = ["disabled"], sd = ["disabled"], cd = {
	key: 2,
	class: "workspace-note"
}, ld = {
	key: 0,
	class: "panel"
}, ud = { class: "panel-head" }, dd = { class: "form-grid phrase-form" }, fd = { class: "field wide" }, pd = ["disabled"], md = { class: "field" }, hd = ["disabled"], gd = ["value"], _d = {
	key: 0,
	class: "field"
}, vd = ["disabled"], yd = ["value"], bd = { class: "field" }, xd = ["disabled"], Sd = ["disabled"], Cd = ["disabled"], wd = ["disabled"], Td = { class: "row form-actions" }, Ed = ["disabled"], Dd = ["disabled"], Od = ["disabled"], kd = {
	key: 1,
	class: "panel"
}, Ad = { class: "panel-head" }, jd = { class: "stats" }, Md = { class: "model-picker" }, Nd = {
	class: "model-choice-grid",
	role: "group",
	"aria-label": "Wake-word models to train"
}, Pd = ["aria-pressed"], Fd = ["aria-pressed"], Id = ["aria-pressed"], Ld = {
	key: 0,
	class: "nano-architecture-panel"
}, Rd = { class: "nano-architecture-heading" }, zd = {
	class: "nano-architecture-grid",
	role: "group",
	"aria-label": "NanoWakeWord architectures"
}, Bd = ["aria-pressed"], Vd = ["aria-pressed"], Hd = {
	key: 1,
	class: "hard-negative-card"
}, Ud = { class: "hard-negative-head" }, Wd = { class: "hard-negative-engines" }, Gd = { class: "field" }, Kd = { class: "wizard-actions" }, qd = ["disabled"], Jd = { class: "workspace-advanced" }, Yd = { class: "row" }, Xd = ["disabled"], Zd = ["disabled"], Qd = ["disabled"], $d = {
	key: 2,
	class: "panel"
}, ef = { class: "workspace-summary" }, tf = { class: "wizard-actions" }, nf = ["disabled"], rf = { class: "hero auto-hero" }, af = { class: "panel" }, of = { class: "toggle-list" }, sf = { class: "form-grid" }, cf = { class: "field" }, lf = { class: "field" }, uf = {
	key: 0,
	class: "field"
}, df = ["value"], ff = { class: "field wide" }, pf = ["value"], mf = { class: "field" }, hf = { class: "panel" }, gf = { class: "form-grid" }, _f = { class: "field" }, vf = { class: "field" }, yf = { class: "model-picker" }, bf = {
	class: "model-choice-grid",
	role: "group",
	"aria-label": "Auto Training wake-word models"
}, xf = ["aria-pressed"], Sf = ["aria-pressed"], Cf = ["aria-pressed"], wf = {
	key: 0,
	class: "nano-architecture-panel"
}, Tf = { class: "nano-architecture-heading" }, Ef = {
	class: "nano-architecture-grid",
	role: "group",
	"aria-label": "Auto Training NanoWakeWord architectures"
}, Df = ["aria-pressed"], Of = ["aria-pressed"], kf = {
	key: 1,
	class: "hard-negative-card"
}, Af = { class: "hard-negative-head" }, jf = { class: "hard-negative-engines" }, Mf = { class: "field" }, Nf = {
	key: 0,
	class: "training-summary-card"
}, Pf = { class: "training-summary-grid" }, Ff = {
	key: 0,
	class: "summary-metrics"
}, If = { class: "stats" }, Lf = { class: "format-value" }, Rf = { class: "format-value" }, zf = { class: "panel" }, Bf = { class: "form-grid" }, Vf = { class: "field wide" }, Hf = { class: "field wide" }, Uf = { class: "link-row" }, Wf = ["disabled"], Gf = ["disabled"], Kf = { class: "toggle-list compact" }, qf = { class: "panel action-panel" }, Jf = { class: "action-grid" }, Yf = ["disabled"], Xf = ["disabled"], Zf = ["disabled"], Qf = ["disabled"], $f = { class: "audit" }, ep = { class: "hero capture-hero" }, tp = { class: "panel" }, np = { class: "panel-head" }, rp = ["disabled"], ip = { class: "stats" }, ap = { class: "panel" }, op = {
	key: 0,
	class: "empty-state"
}, sp = {
	key: 1,
	class: "audio-list"
}, cp = {
	key: 0,
	class: "meta-row"
}, lp = ["onUpdate:modelValue"], up = ["value"], dp = {
	key: 1,
	class: "transcript"
}, fp = {
	key: 2,
	class: "transcript"
}, pp = ["src"], mp = ["disabled", "onClick"], hp = ["disabled", "onClick"], gp = ["disabled", "onClick"], _p = {
	key: 0,
	class: "hero samples-hero"
}, vp = { class: "pill hero-pill" }, yp = { class: "panel" }, bp = { class: "panel-head sample-head" }, xp = { class: "segment-control" }, Sp = { class: "row toolbar" }, Cp = ["disabled"], wp = ["disabled"], Tp = ["disabled"], Ep = {
	key: 0,
	class: "empty-state"
}, Dp = {
	key: 1,
	class: "audio-list compact-list"
}, Op = { class: "row" }, kp = {
	key: 0,
	class: "pill warning"
}, Ap = {
	key: 0,
	class: "transcript"
}, jp = {
	key: 1,
	class: "transcript"
}, Mp = ["src"], Np = ["onClick"], Pp = ["onClick"], Fp = ["disabled", "onClick"], Ip = {
	key: 2,
	class: "pagination"
}, Lp = ["disabled"], Rp = ["disabled"], zp = { class: "panel" }, Bp = { class: "dropzone" }, Vp = ["disabled"], Hp = { class: "progress-card" }, Up = { class: "progress-track" }, Wp = {
	key: 1,
	class: "wizard-actions"
}, Gp = { class: "hero data-hero" }, Kp = { class: "pill hero-pill" }, qp = {
	key: 0,
	class: "panel storage-location-panel"
}, Jp = { class: "panel-head" }, Yp = ["disabled"], Xp = { class: "storage-location" }, Zp = {
	key: 0,
	class: "pill success"
}, Qp = {
	key: 1,
	class: "pill neutral"
}, $p = {
	key: 0,
	class: "data-warning"
}, em = { class: "panel" }, tm = { class: "panel-head" }, nm = ["disabled"], rm = { class: "stats" }, im = { class: "format-value" }, am = {
	key: 0,
	class: "data-warning"
}, om = { class: "panel-head" }, sm = { class: "number" }, cm = { class: "data-list" }, lm = { class: "data-copy" }, um = { class: "data-title" }, dm = {
	key: 0,
	class: "data-note"
}, fm = { class: "data-usage" }, pm = ["disabled", "onClick"], mm = {
	key: 1,
	class: "panel empty-state"
}, hm = {
	key: 1,
	class: "workspace-footer"
}, gm = { class: "row" }, _m = {
	class: "modal link-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Link Tater"
}, vm = { class: "modal-head" }, ym = {
	key: 0,
	class: "link-success"
}, bm = {
	key: 1,
	class: "stack"
}, xm = { class: "field" }, Sm = { class: "field" }, Cm = ["disabled"], wm = 50, Tm = /* @__PURE__ */ fr({
	__name: "TrainerApp",
	setup(e) {
		let t = /* @__PURE__ */ F(null), n = /* @__PURE__ */ F(null), r = /* @__PURE__ */ F(!0), i = /* @__PURE__ */ F(""), a = /* @__PURE__ */ F(""), o = /* @__PURE__ */ F(!1), s = [
			{
				id: "home",
				label: "Start",
				icon: "home"
			},
			{
				id: "firmware",
				label: "Wake words",
				icon: "wave"
			},
			{
				id: "captured",
				label: "Captures",
				icon: "inbox"
			},
			{
				id: "auto",
				label: "Auto training",
				icon: "refresh"
			}
		], c = q(() => J.wakeWordLibrary.find((e) => e.safe_word === J.selectedWord)), l = q(() => [
			...J.trainMicroWakeWord ? ["MicroWakeWord"] : [],
			...J.trainOpenWakeWord ? ["openWakeWord"] : [],
			...J.trainNanoWakeWordCnn ? ["NanoWakeWord CNN"] : [],
			...J.trainNanoWakeWordGru ? ["NanoWakeWord GRU"] : []
		]), u = q(() => {
			let e = String(J.training.safe_word || J.training.summary?.safe_word || ""), t = J.wakeWordLibrary.find((t) => t.safe_word === e)?.phrase;
			return (J.training.running ? t || J.session.raw_phrase : J.training.summary?.wake_word || t) || e || J.session.raw_phrase || "Wake word";
		}), d = q(() => {
			let e = J.samplePage[J.sampleBucket];
			return Rs.value.slice(e * wm, (e + 1) * wm);
		}), f = q(() => Math.max(1, Math.ceil(Rs.value.length / wm))), p = q(() => J.auto.state || {}), m = q(() => J.auto.runtime || {}), h = q(() => {
			let e = p.value, t = [];
			return e.last_review_result && t.push(`Last review: ${String(e.last_review_result).replaceAll("_", " ")}`), e.last_review_file && t.push(String(e.last_review_file)), e.last_review_transcript && t.push(`STT: “${e.last_review_transcript}”`), e.last_review_error && t.push(`Error: ${e.last_review_error}`), e.last_stt_engine && t.push(`STT engine: ${String(e.last_stt_engine).replaceAll("_", " ")}`), e.last_notify_at && t.push(e.last_notify_error ? `Publish failed: ${e.last_notify_error}` : `Wake word published ${Oc(e.last_notify_at)}`), t.join(" · ") || "No automatic review has run yet.";
		}), g = q(() => J.training.running ? {
			text: "Training running",
			tone: "warning"
		} : J.training.exit_code === 0 ? {
			text: "Training finished",
			tone: "success"
		} : J.training.exit_code === null ? {
			text: "Not started",
			tone: "neutral"
		} : {
			text: `Exit ${J.training.exit_code}`,
			tone: "error"
		}), _ = q(() => m.value.review_running ? {
			text: `Transcribing ${m.value.review_file || "wake"}`,
			tone: "warning"
		} : J.training.running && J.auto.config?.enabled ? {
			text: "Training running",
			tone: "warning"
		} : J.auto.config?.enabled ? {
			text: "Enabled",
			tone: "success"
		} : {
			text: "Disabled",
			tone: "neutral"
		}), v = q(() => J.training.log_lines?.length ? J.training.log_lines : ["No training output yet."]), y = q(() => J.training.progress || null), b = q(() => {
			let e = String(y.value?.stage || "pipeline").split(":", 1)[0];
			return {
				starting: "Getting ready",
				datasets: "Training data",
				tts: "Speech generation",
				features: "Feature building",
				model: "Model training",
				boundary: "Wake boundary",
				calibration: "Calibration",
				package: "Packaging",
				openwakeword: "openWakeWord",
				nanowakeword: "NanoWakeWord",
				complete: "Finished",
				failed: "Needs attention",
				worker: "Pipeline progress",
				pipeline: "Live pipeline"
			}[e] || e.replaceAll("-", " ");
		}), x = q(() => {
			let e = y.value?.percent;
			if (e == null) return null;
			let t = Number(e);
			return Number.isFinite(t) ? Math.max(0, Math.min(100, t)) : null;
		}), S = q(() => {
			let e = y.value;
			if (!e) return [];
			let t = [], n = Number(e.current), r = Number(e.total), i = String(e.unit || "items");
			return Number.isFinite(n) && Number.isFinite(r) && r > 0 ? t.push(`${Math.round(n).toLocaleString()} / ${Math.round(r).toLocaleString()} ${i}`) : Number.isFinite(n) && n > 0 && t.push(`${Math.round(n).toLocaleString()} ${i}`), e.elapsed && t.push(`Elapsed ${e.elapsed}`), e.eta && t.push(`About ${e.eta} left`), e.rate_text ? t.push(String(e.rate_text)) : Number(e.rate) > 0 && t.push(`${Number(e.rate).toLocaleString()} ${i}/sec`), t;
		}), C = q(() => {
			let e = String(J.session.safe_word || ""), t = J.wakeWords.filter((e) => !e.dual_model);
			return t.find((t) => String(t.key || "") === e) || t[0];
		}), w = q(() => (Array.isArray(J.training.summary?.models) ? J.training.summary.models : []).find((e) => e.failure_kind === "quality_rejected" && String(e.id || "").startsWith("nano_"))), ee = q(() => J.wakeWordLibrary.find((e) => e.safe_word === J.session.safe_word)), te = q(() => {
			let e = /* @__PURE__ */ new Map();
			for (let t of J.managedData.items || []) {
				let n = e.get(t.category) || [];
				n.push(t), e.set(t.category, n);
			}
			return Array.from(e, ([e, t]) => ({
				name: e,
				items: t
			}));
		}), ne = q(() => J.trainNanoWakeWordCnn || J.trainNanoWakeWordGru), T = q(() => J.autoForm.train_nanowakeword_cnn || J.autoForm.train_nanowakeword_gru), re = q(() => [
			...J.trainMicroWakeWord ? ["MWW"] : [],
			...J.trainOpenWakeWord ? ["OWW"] : [],
			...ne.value ? ["NWW"] : []
		]), E = q(() => [
			...J.autoForm.train_microwakeword ? ["MWW"] : [],
			...J.autoForm.train_openwakeword ? ["OWW"] : [],
			...T.value ? ["NWW"] : []
		]);
		Nn([() => J.language, () => J.ttsMode], Js), Nn(() => J.toast.serial, () => window.setTimeout(() => {
			J.toast.message = "";
		}, 4500)), Nn(v, async () => {
			r.value && (await hn(), r.value && n.value && (n.value.scrollTop = n.value.scrollHeight));
		}), Nn(() => J.activeView, async (e) => {
			await hn(), window.scrollTo(0, 0), e === "training" && oe();
		}), Nn(() => J.consoleOpen, (e) => {
			e && (J.activeView = "training", J.consoleOpen = !1);
		}), Dr(() => {
			Ec(), document.addEventListener("keydown", ie);
		}), Ar(() => {
			Dc(), document.removeEventListener("keydown", ie);
		});
		function ie(e) {
			e.key === "Escape" && (J.taterLinkOpen = !1, J.trimItem = null);
		}
		function ae() {
			let e = n.value;
			if (!e) return;
			let t = e.scrollHeight - e.clientHeight - e.scrollTop;
			r.value = t <= 32;
		}
		function oe() {
			let e = n.value;
			e && (r.value = !0, e.scrollTop = e.scrollHeight);
		}
		function se(e) {
			J.activeView = e, document.querySelectorAll("details.workspace-more[open]").forEach((e) => e.removeAttribute("open")), (e === "home" ? dc(!0) : e === "auto" ? ac(!1) : e === "captured" ? Xs() : e === "samples" ? Ys() : e === "firmware" ? Promise.all([uc(), dc()]) : e === "data" ? mc() : Promise.resolve()).catch((e) => Z(e instanceof Error ? e.message : "Refresh failed.", "error"));
		}
		function D(e) {
			J.sampleBucket = e;
		}
		function ce(e, t) {
			J.trimBucket = t, J.trimItem = e;
		}
		function le() {
			ne.value ? (J.trainNanoWakeWordCnn = !1, J.trainNanoWakeWordGru = !1) : J.trainNanoWakeWordCnn = !0;
		}
		function ue() {
			T.value ? (J.autoForm.train_nanowakeword_cnn = !1, J.autoForm.train_nanowakeword_gru = !1) : J.autoForm.train_nanowakeword_cnn = !0;
		}
		function de(e) {
			return new Set(String(e || "").split(/[\n,]+/).map((e) => e.trim().toLocaleLowerCase()).filter(Boolean)).size;
		}
		function pe() {
			i.value = J.autoForm.tater_url || "http://127.0.0.1:8501", a.value = "", o.value = !1, J.taterLinkOpen = !0, hn(() => document.querySelector("#pairing-code")?.focus());
		}
		function me() {
			let e = a.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
			a.value = e.length > 4 ? `${e.slice(0, 4)}-${e.slice(4)}` : e;
		}
		async function he() {
			if (!i.value.trim() || !a.value.trim()) {
				Z("Tater address and pairing code are required.", "warning");
				return;
			}
			o.value = await cc(i.value, a.value);
		}
		function ge(e) {
			let t = [];
			return e.source_device && t.push(String(e.source_device)), e.wake_word && t.push(String(e.wake_word)), e.max_probability !== null && e.max_probability !== void 0 && t.push(`max ${e.max_probability}`), e.average_probability !== null && e.average_probability !== void 0 && t.push(`avg ${e.average_probability}`), e.detection_profile && t.push(`profile ${String(e.detection_profile).replaceAll("_", " ")}`), e.auto_review_status && t.push(`auto ${String(e.auto_review_status).replaceAll("_", " ")}`), e.vad_max_probability !== null && e.vad_max_probability !== void 0 && t.push(`VAD ${e.vad_max_probability}`), t;
		}
		function _e(e) {
			let t = [];
			e.original_name && e.original_name !== e.saved_as && t.push(`From ${e.original_name}`);
			let n = Oc(e.reviewed_at || e.received_at || e.created_at);
			return n && t.push(`Saved ${n}`), e.message && t.push(String(e.message)), e.auto_negative && t.push("Auto-reviewed false positive"), e.auto_positive && t.push("Auto-promoted close miss"), t.join(" · ") || "Training sample";
		}
		function ve(e) {
			let t = e.trim().toLowerCase();
			return /^(✓|✅)|success|finished/.test(t) ? "success" : /^(✗|❌)|error|failed|traceback/.test(t) ? "error" : /^(⚠|warning)/.test(t) ? "warning" : /^={4,}|^-----|^=====/.test(t) ? "heading" : "";
		}
		function ye(e) {
			J.selectedWord = e.safe_word, J.activeView = "word";
		}
		async function be() {
			if (J.training.running || Y("session")) {
				Z("Wait for the active training run before creating another word.", "warning");
				return;
			}
			J.session.safe_word && (await Ks(), J.session.safe_word) || (J.phrase = "", J.selectedWord = "", J.wizardRetrain = !1, J.wizardStep = 0, J.activeView = "trainer");
		}
		async function xe(e) {
			return J.session.safe_word === e.safe_word ? !0 : J.training.running ? (Z("Wait for training before changing the recording workspace.", "warning"), !1) : J.session.safe_word && (await Ks(), J.session.safe_word) ? !1 : (await fc(e), J.session.safe_word === e.safe_word);
		}
		async function Se(e) {
			await xe(e) && (J.selectedWord = e.safe_word, J.wizardRetrain = !0, J.wizardStep = 2, J.activeView = "trainer");
		}
		async function Ce(e) {
			await xe(e) && (J.selectedWord = e.safe_word, se("samples"));
		}
		async function we() {
			if (!J.session.safe_word) {
				let e = J.wakeWordLibrary.find((e) => e.phrase.trim().toLocaleLowerCase() === J.phrase.trim().toLocaleLowerCase());
				if (e) {
					ye(e), Z("This phrase is already saved. Open its results or train a new version.", "warning");
					return;
				}
				if (await Gs(), !J.session.safe_word) return;
			}
			await Ys(!0), J.wizardStep = 1;
		}
		async function Te() {
			await Ks(), J.session.safe_word || (J.wizardStep = 0);
		}
		function A() {
			let e = String(J.training.safe_word || J.training.summary?.safe_word || J.session.safe_word || ""), t = J.wakeWordLibrary.find((t) => t.safe_word === e);
			t ? ye(t) : se("firmware");
		}
		function Ee(e) {
			let t = e.split(":", 1)[0];
			return {
				starting: "Getting ready",
				datasets: "Training data",
				tts: "Speech generation",
				features: "Feature building",
				model: "Model training",
				boundary: "Wake boundary",
				calibration: "Calibration",
				package: "JSON & models",
				openwakeword: "openWakeWord",
				nanowakeword: "NanoWakeWord",
				complete: "Finished",
				failed: "Needs attention"
			}[t] || t.replaceAll("-", " ");
		}
		return (e, m) => (V(), H("div", Su, [
			U("header", Cu, [U("div", wu, [m[66] ||= U("span", { class: "tater-wordmark" }, "Tater", -1), U("div", null, [m[65] ||= U("h1", null, "Wake Word Trainer", -1), U("div", Tu, [
				U("span", null, k("Apple Silicon · local trainer"), 1),
				U("small", null, k("v47.10")),
				U("small", { title: I(J).liveState === "live" ? "Updates arrive automatically" : "Updates retry automatically when the connection returns" }, k({
					live: "Live",
					connecting: "Connecting…",
					reconnecting: "Reconnecting…",
					polling: "Auto refresh",
					paused: "Paused"
				}[I(J).liveState]), 9, Eu)
			])])]), U("nav", Du, [
				(V(), H(B, null, z(s, (e) => U("button", {
					key: e.id,
					type: "button",
					"aria-current": I(J).activeView === e.id || e.id === "firmware" && I(J).activeView === "word" ? "page" : void 0,
					onClick: (t) => se(e.id)
				}, [
					W($, { name: e.icon }, null, 8, ["name"]),
					G(k(e.label), 1),
					e.id === "captured" && I(J).captured.captured_count ? (V(), H("span", ku, k(I(J).captured.captured_count), 1)) : K("", !0)
				], 8, Ou)), 64)),
				I(Ls) ? (V(), H("button", {
					key: 0,
					type: "button",
					"aria-current": I(J).activeView === "training" ? "page" : void 0,
					onClick: m[0] ||= (e) => se("training")
				}, [W($, { name: "activity" }), m[67] ||= G("Training", -1)], 8, Au)) : K("", !0),
				U("details", ju, [m[68] ||= U("summary", null, "More", -1), U("div", null, [
					I(J).session.safe_word ? (V(), H("button", {
						key: 0,
						type: "button",
						onClick: m[1] ||= (e) => se("trainer")
					}, "Resume setup · " + k(I(J).session.raw_phrase), 1)) : K("", !0),
					U("button", {
						type: "button",
						onClick: m[2] ||= (e) => se("samples")
					}, "Samples"),
					U("button", {
						type: "button",
						onClick: m[3] ||= (e) => se("data")
					}, "Data & storage"),
					U("button", {
						type: "button",
						onClick: pe
					}, k(I(zs) ? "Tater connection" : "Link Tater"), 1)
				])])
			])]),
			I(J).training.running && I(J).activeView !== "training" ? (V(), H("div", Mu, [U("span", Nu, [
				W($, { name: "activity" }),
				G("Training “" + k(u.value) + "” · " + k(b.value), 1),
				x.value === null ? K("", !0) : (V(), H(B, { key: 0 }, [G(" · " + k(x.value.toFixed(1)) + "%", 1)], 64))
			]), U("button", {
				type: "button",
				onClick: m[4] ||= (e) => se("training")
			}, [m[69] ||= G("Open training", -1), W($, { name: "external" })])])) : K("", !0),
			U("main", Pu, [I(J).initialized ? (V(), H(B, { key: 1 }, [I(J).activeView === "trainer" ? (V(), H(B, { key: 0 }, [
				U("button", {
					type: "button",
					class: "workspace-back",
					onClick: m[5] ||= (e) => I(J).activeView = "home"
				}, [W($, { name: "back" }), m[71] ||= G("Back to start", -1)]),
				U("header", Iu, [U("div", null, [
					U("span", Lu, k(I(J).wizardRetrain ? "Improve an existing word" : "New wake word"), 1),
					U("h2", null, k(I(J).wizardRetrain ? `Train a new version of “${I(J).phrase}”` : "Create a wake word"), 1),
					m[72] ||= U("p", null, "Phrase → recordings → models → review.", -1)
				]), I(J).session.safe_word ? (V(), H("span", Ru, k(I(J).session.raw_phrase), 1)) : K("", !0)]),
				U("div", zu, [(V(), H(B, null, z([
					"Phrase",
					"Recordings",
					"Models",
					"Review"
				], (e, t) => U("button", {
					key: e,
					type: "button",
					"aria-current": I(J).wizardStep === t ? "step" : void 0,
					disabled: t > 0 && !I(J).session.safe_word || t === 3 && !l.value.length,
					onClick: (e) => I(J).wizardStep = t
				}, [U("b", null, k(t < I(J).wizardStep ? "✓" : t + 1), 1), G(k(e), 1)], 8, Bu)), 64))])
			], 64)) : K("", !0), I(J).activeView === "home" ? (V(), aa(Cl, {
				key: 1,
				onCreate: be,
				onOpen: ye
			})) : I(J).activeView === "firmware" || I(J).activeView === "word" && !c.value ? (V(), aa(Cl, {
				key: 2,
				library: "",
				onCreate: be,
				onOpen: ye
			})) : I(J).activeView === "word" && c.value ? (V(), aa(xu, {
				key: 3,
				word: c.value,
				onRetrain: Se,
				onRecordings: Ce
			}, null, 8, ["word"])) : I(J).activeView === "training" ? (V(), H(B, { key: 4 }, [
				U("button", {
					type: "button",
					class: "workspace-back",
					onClick: A
				}, [W($, { name: "back" }), m[73] ||= G("Back to wake word", -1)]),
				U("header", Vu, [U("div", null, [
					m[74] ||= U("span", { class: "eyebrow" }, "Training workspace", -1),
					U("h2", null, k(I(J).training.running ? "Training" : "Training results for") + " “" + k(u.value) + "”", 1),
					U("p", null, k(I(J).training.running ? "The current published version and its JSON links stay available while this run trains." : "Review the report, view saved results, or train another version."), 1)
				]), U("span", { class: O(["pill", g.value.tone]) }, k(g.value.text), 3)]),
				U("div", Hu, [U("section", Uu, [
					U("header", Wu, [U("div", null, [U("h3", null, k(b.value), 1), U("p", null, k(y.value?.label || (I(J).training.running ? "Waiting for training output…" : "No stage details available.")), 1)])]),
					y.value ? (V(), H("div", {
						key: 0,
						class: O(["console-progress", {
							indeterminate: x.value === null && !["complete", "failed"].includes(y.value.status || ""),
							complete: y.value.status === "complete",
							failed: y.value.status === "failed"
						}])
					}, [
						U("div", Gu, [U("div", null, [m[75] ||= U("span", null, "Current stage progress", -1), U("strong", null, k(y.value.label || "Working…"), 1)]), U("b", null, k(y.value.status === "failed" ? "Failed" : y.value.status === "complete" ? "Done" : x.value === null ? "Working" : `${x.value.toFixed(1)}%`), 1)]),
						U("div", {
							class: "console-progress-track",
							role: "progressbar",
							"aria-label": "Current training stage progress",
							"aria-valuenow": x.value ?? void 0,
							"aria-valuemin": "0",
							"aria-valuemax": "100"
						}, [U("i", { style: fe({ width: ["complete", "failed"].includes(y.value.status || "") ? "100%" : x.value === null ? "38%" : `${x.value}%` }) }, null, 4)], 8, Ku),
						S.value.length ? (V(), H("div", qu, [(V(!0), H(B, null, z(S.value, (e) => (V(), H("span", { key: e }, k(e), 1))), 128))])) : K("", !0)
					], 2)) : K("", !0),
					U("div", Ju, [(V(!0), H(B, null, z(I(J).trainingStages, (e, t) => (V(), H("article", {
						key: `${t}-${e.stage}`,
						class: O(e.status)
					}, [
						W($, { name: e.status === "complete" ? "check" : e.status === "running" ? "activity" : "circle" }, null, 8, ["name"]),
						U("div", null, [U("strong", null, k(Ee(e.stage)), 1), U("small", null, k(e.label), 1)]),
						U("span", { class: O(["pill", e.status === "complete" ? "success" : e.status === "failed" ? "error" : ""]) }, k(e.status === "complete" ? "Done" : e.status === "failed" ? "Needs attention" : e.status === "running" ? "Running" : e.status), 3)
					], 2))), 128)), I(J).trainingStages.length ? K("", !0) : (V(), H("p", Yu, "Stage details appear as the trainer reports them."))])
				]), U("aside", Xu, [U("section", Zu, [m[81] ||= U("h3", null, "This training run", -1), U("dl", Qu, [
					U("div", null, [m[76] ||= U("dt", null, "Wake word", -1), U("dd", null, k(u.value), 1)]),
					U("div", null, [m[77] ||= U("dt", null, "Trainer", -1), U("dd", null, k("Apple Silicon · local trainer"), 1)]),
					U("div", null, [m[78] ||= U("dt", null, "Recordings", -1), U("dd", null, k(I(Ns)) + " personal · " + k(I(Ps)) + " negatives", 1)]),
					I(J).trainingTargets.length ? (V(), H("div", $u, [m[79] ||= U("dt", null, "Requested models", -1), U("dd", null, k(I(J).trainingTargets.join(", ")), 1)])) : K("", !0),
					m[80] ||= U("div", null, [U("dt", null, "Published JSON"), U("dd", null, "Existing versions remain available")], -1)
				])]), m[82] ||= pa("<section class=\"panel\"><h3>When it is ready</h3><p class=\"workspace-note\">Validation results, a saved version, and its JSON links appear on the word’s page.</p><div class=\"row\"><span class=\"pill\">Bundle JSON</span><span class=\"pill\">ESPHome JSON with MicroWakeWord</span></div></section>", 1)])]),
				U("section", ed, [U("header", td, [m[83] ||= U("div", null, [U("h3", null, "Training log"), U("p", null, "Training continues while you browse other pages.")], -1), r.value ? K("", !0) : (V(), H("button", {
					key: 0,
					type: "button",
					class: "console-follow",
					onClick: oe
				}, "Jump to latest"))]), U("pre", {
					ref_key: "consoleLog",
					ref: n,
					class: "console-log",
					onScrollPassive: ae
				}, [(V(!0), H(B, null, z(v.value, (e, t) => (V(), H("span", {
					key: `${t}-${e}`,
					class: O(ve(e))
				}, k(e), 3))), 128))], 544)]),
				!I(J).training.running && I(J).training.summary?.models?.length ? (V(), H("section", nd, [U("header", null, [
					U("div", { class: O(["summary-mark", I(J).training.summary.tone]) }, k(I(J).training.summary.tone === "success" ? "✓" : I(J).training.summary.tone === "error" ? "×" : "!"), 3),
					U("div", null, [
						m[84] ||= U("span", { class: "eyebrow" }, "Last completed run", -1),
						U("h4", null, k(I(J).training.summary.wake_word), 1),
						U("p", null, [G(k(I(J).training.summary.message), 1), I(J).training.summary.completed_at ? (V(), H(B, { key: 0 }, [G(" · " + k(I(Oc)(I(J).training.summary.completed_at)), 1)], 64)) : K("", !0)])
					]),
					U("span", { class: O(["pill", I(J).training.summary.tone]) }, k(I(J).training.summary.overall), 3)
				]), U("div", rd, [(V(!0), H(B, null, z(I(J).training.summary.models, (e) => (V(), H("article", {
					key: e.id,
					class: O(e.tone)
				}, [
					U("header", null, [U("div", null, [U("span", null, k(e.id.replaceAll("_", " ")), 1), U("strong", null, k(e.label), 1)]), U("b", null, k(e.rating), 1)]),
					U("p", null, k(e.message), 1),
					e.metrics?.length ? (V(), H("div", id, [(V(!0), H(B, null, z(e.metrics, (e) => (V(), H("span", { key: `${e.label}-${e.value}` }, [U("small", null, k(e.label), 1), U("strong", null, k(e.value), 1)]))), 128))])) : K("", !0)
				], 2))), 128))])])) : K("", !0),
				U("div", ad, [I(J).training.running ? (V(), H("button", {
					key: 0,
					type: "button",
					disabled: I(Y)("session"),
					onClick: m[6] ||= (...e) => I(Ks) && I(Ks)(...e)
				}, [W($, { name: "stop" }), m[85] ||= G("Stop training", -1)], 8, od)) : I(J).session.safe_word ? (V(), H("button", {
					key: 1,
					type: "button",
					disabled: I(Y)("session"),
					onClick: Te
				}, "End setup session", 8, sd)) : (V(), H("span", cd, "Every saved version stays in your library.")), U("button", {
					type: "button",
					class: "primary",
					onClick: A
				}, [G(k(I(J).training.running ? "View current word" : "View results & JSON"), 1), W($, { name: "next" })])])
			], 64)) : I(J).activeView === "trainer" && I(J).wizardStep !== 1 ? (V(), H(B, { key: 5 }, [I(J).wizardStep === 0 ? (V(), H("section", ld, [
				U("header", ud, [
					m[86] ||= U("div", { class: "number" }, "1", -1),
					m[87] ||= U("div", null, [U("h3", null, "Phrase + voice"), U("p", null, "The phrase and voice route lock while a session is active.")], -1),
					U("span", { class: O(["pill", I(J).session.safe_word ? "success" : ""]) }, k(I(J).session.safe_word ? `Session · ${I(J).session.safe_word}` : "No session"), 3)
				]),
				U("div", dd, [
					U("label", fd, [m[88] ||= U("span", null, "Wake phrase", -1), R(U("input", {
						"onUpdate:modelValue": m[7] ||= (e) => I(J).phrase = e,
						type: "text",
						placeholder: "e.g. \"hey tater\"",
						disabled: !!I(J).session.safe_word || I(Y)("session"),
						onKeyup: ls(we, ["enter"])
					}, null, 40, pd), [[Qo, I(J).phrase]])]),
					U("label", md, [
						m[89] ||= U("span", null, "Language", -1),
						R(U("select", {
							"onUpdate:modelValue": m[8] ||= (e) => I(J).language = e,
							disabled: !!I(J).session.safe_word || I(Y)("session")
						}, [(V(!0), H(B, null, z(I(J).languages, (e) => (V(), H("option", {
							key: e.code,
							value: e.code
						}, k(e.label), 9, gd))), 128))], 8, hd), [[ts, I(J).language]]),
						U("small", null, k(I(Is)), 1)
					]),
					I(J).language === "en" && I(J).ttsMode !== "piper" ? (V(), H("label", _d, [
						m[90] ||= U("span", null, "English accent emphasis", -1),
						R(U("select", {
							"onUpdate:modelValue": m[9] ||= (e) => I(J).englishAccent = e,
							disabled: !!I(J).session.safe_word || I(Y)("session")
						}, [(V(!0), H(B, null, z(I(J).englishAccents, (e) => (V(), H("option", {
							key: e.code,
							value: e.code
						}, k(e.label), 9, yd))), 128))], 8, vd), [[ts, I(J).englishAccent]]),
						m[91] ||= U("small", null, "Qwen emphasizes this accent; MOSS carries it through accepted references. OmniVoice and Piper keep broad English coverage.", -1)
					])) : K("", !0),
					U("label", bd, [
						m[92] ||= U("span", null, "TTS source", -1),
						R(U("select", {
							"onUpdate:modelValue": m[10] ||= (e) => I(J).ttsMode = e,
							disabled: !!I(J).session.safe_word || I(Y)("session")
						}, [
							U("option", {
								value: "hybrid",
								disabled: !I(J).languages.find((e) => e.code === I(J).language)?.engines?.includes("piper")
							}, "Four-provider ensemble · recommended", 8, Sd),
							U("option", {
								value: "modern",
								disabled: !I(J).languages.find((e) => e.code === I(J).language)?.engines?.some((e) => e !== "piper")
							}, "Modern only · no Piper", 8, Cd),
							U("option", {
								value: "piper",
								disabled: !I(J).languages.find((e) => e.code === I(J).language)?.engines?.includes("piper")
							}, "Piper only · legacy", 8, wd)
						], 8, xd), [[ts, I(J).ttsMode]]),
						m[93] ||= U("small", null, "Models download once and stay cached.", -1)
					])
				]),
				U("div", Td, [
					I(J).session.safe_word ? (V(), H("button", {
						key: 1,
						type: "button",
						class: "button danger",
						disabled: I(Y)("session"),
						onClick: Te
					}, k(I(Y)("session") ? "Stopping…" : I(J).training.running ? "Stop training + end session" : "End setup session"), 9, Dd)) : (V(), H("button", {
						key: 0,
						type: "button",
						class: "button primary",
						disabled: I(Y)("session") || !I(J).phrase.trim(),
						onClick: we
					}, k(I(Y)("session") ? "Starting…" : "Continue"), 9, Ed)),
					I(J).session.safe_word ? (V(), H("button", {
						key: 2,
						type: "button",
						class: "primary",
						onClick: we
					}, "Continue to recordings")) : K("", !0),
					U("button", {
						type: "button",
						disabled: !I(J).phrase.trim(),
						onClick: m[11] ||= (...e) => I(qs) && I(qs)(...e)
					}, "System preview", 8, Od)
				])
			])) : I(J).wizardStep === 2 ? (V(), H("section", kd, [
				U("header", Ad, [
					m[94] ||= U("div", { class: "number" }, "3", -1),
					m[95] ||= U("div", null, [U("h3", null, "Train wake word"), U("p", null, "Personal positives and reviewed false-wake negatives are automatically included.")], -1),
					U("span", { class: O(["pill", g.value.tone]) }, k(g.value.text), 3)
				]),
				U("div", jd, [
					U("article", null, [m[96] ||= U("span", null, "Positive samples", -1), U("strong", null, k(I(Ns)), 1)]),
					U("article", null, [m[97] ||= U("span", null, "Negative samples", -1), U("strong", null, k(I(Ps)), 1)]),
					m[98] ||= U("article", null, [U("span", null, "Training format"), U("strong", { class: "format-value" }, "16 kHz · mono · WAV")], -1)
				]),
				U("div", Md, [
					U("div", Nd, [
						U("button", {
							type: "button",
							class: O(["model-choice", { active: I(J).trainMicroWakeWord }]),
							"aria-pressed": I(J).trainMicroWakeWord,
							onClick: m[12] ||= (e) => I(J).trainMicroWakeWord = !I(J).trainMicroWakeWord
						}, [...m[99] ||= [
							U("i", null, "MWW", -1),
							U("span", null, [U("strong", null, "MicroWakeWord"), U("small", null, "TFLite for ESP32 satellites and ESPHome.")], -1),
							U("b", null, "✓", -1)
						]], 10, Pd),
						U("button", {
							type: "button",
							class: O(["model-choice", { active: I(J).trainOpenWakeWord }]),
							"aria-pressed": I(J).trainOpenWakeWord,
							onClick: m[13] ||= (e) => I(J).trainOpenWakeWord = !I(J).trainOpenWakeWord
						}, [...m[100] ||= [
							U("i", null, "OWW", -1),
							U("span", null, [U("strong", null, "openWakeWord"), U("small", null, "ONNX for satellites with OWW support.")], -1),
							U("b", null, "✓", -1)
						]], 10, Fd),
						U("button", {
							type: "button",
							class: O(["model-choice", { active: ne.value }]),
							"aria-pressed": ne.value,
							onClick: le
						}, [...m[101] ||= [
							U("i", null, "NWW", -1),
							U("span", null, [U("strong", null, "NanoWakeWord"), U("small", null, "ONNX with selectable CNN and GRU builds.")], -1),
							U("b", null, "✓", -1)
						]], 10, Id)
					]),
					ne.value ? (V(), H("div", Ld, [U("div", Rd, [m[102] ||= U("div", null, [U("strong", null, "NanoWakeWord architectures"), U("small", null, "Select either architecture or train both.")], -1), U("span", null, k(I(J).trainNanoWakeWordCnn && I(J).trainNanoWakeWordGru ? "2 selected" : "1 selected"), 1)]), U("div", zd, [U("button", {
						type: "button",
						class: O(["architecture-choice", { active: I(J).trainNanoWakeWordCnn }]),
						"aria-pressed": I(J).trainNanoWakeWordCnn,
						onClick: m[14] ||= (e) => I(J).trainNanoWakeWordCnn = !I(J).trainNanoWakeWordCnn
					}, [...m[103] ||= [
						U("i", null, "CNN", -1),
						U("span", null, [U("strong", null, "CNN"), U("small", null, "Recommended default with stronger recall.")], -1),
						U("b", null, "✓", -1)
					]], 10, Bd), U("button", {
						type: "button",
						class: O(["architecture-choice", { active: I(J).trainNanoWakeWordGru }]),
						"aria-pressed": I(J).trainNanoWakeWordGru,
						onClick: m[15] ||= (e) => I(J).trainNanoWakeWordGru = !I(J).trainNanoWakeWordGru
					}, [...m[104] ||= [
						U("i", null, "GRU", -1),
						U("span", null, [U("strong", null, "GRU"), U("small", null, "Recurrent model tuned for fewer false triggers.")], -1),
						U("b", null, "✓", -1)
					]], 10, Vd)])])) : K("", !0),
					re.value.length ? (V(), H("section", Hd, [
						U("header", Ud, [
							m[105] ||= U("div", { class: "hard-negative-icon" }, "−", -1),
							m[106] ||= U("div", null, [
								U("span", { class: "eyebrow" }, "Optional precision tuning"),
								U("strong", null, "Hard negative training"),
								U("small", null, "Teach the selected models which similar-sounding phrases must not wake the assistant.")
							], -1),
							U("b", null, k(de(I(J).generatedNegativePhrases) ? `${de(I(J).generatedNegativePhrases)} phrase${de(I(J).generatedNegativePhrases) === 1 ? "" : "s"}` : "Optional"), 1)
						]),
						U("div", Wd, [m[107] ||= U("span", null, "Applied to", -1), (V(), H(B, null, z([
							"MWW",
							"OWW",
							"NWW"
						], (e) => U("i", {
							key: e,
							class: O({ active: re.value.includes(e) })
						}, k(e), 3)), 64))]),
						U("label", Gd, [
							m[108] ||= U("span", null, "Phrases to reject", -1),
							R(U("textarea", {
								"onUpdate:modelValue": m[16] ||= (e) => I(J).generatedNegativePhrases = e,
								rows: "4",
								placeholder: "hey later\nhey waiter\nokay tater"
							}, null, 512), [[Qo, I(J).generatedNegativePhrases]]),
							m[109] ||= U("small", null, "One phrase per line. Each selected engine synthesizes and validates these near-misses alongside reviewed recordings and its broad negative data.", -1)
						]),
						m[110] ||= U("footer", null, [U("span", null, "Unchanged phrases reuse their cached audio on later runs."), U("strong", null, "The actual wake phrase is ignored automatically.")], -1)
					])) : K("", !0)
				]),
				U("div", Kd, [U("button", {
					type: "button",
					onClick: m[17] ||= (e) => I(J).wizardStep = 1
				}, "Back to recordings"), U("button", {
					type: "button",
					class: "primary",
					disabled: !I(J).session.safe_word || !l.value.length || I(J).training.running,
					onClick: m[18] ||= (e) => I(J).wizardStep = 3
				}, [m[111] ||= G("Review training", -1), W($, { name: "next" })], 8, qd)]),
				U("details", Jd, [m[112] ||= U("summary", null, "Additional training actions", -1), U("div", Yd, [
					U("button", {
						type: "button",
						class: "boundary-button",
						disabled: !ee.value?.boundary_only_ready || I(J).training.running || I(Y)("boundary-start"),
						onClick: m[19] ||= (...e) => I(yc) && I(yc)(...e)
					}, k(I(Y)("boundary-start") ? "Starting boundary…" : "Retrain boundary only"), 9, Xd),
					C.value ? (V(), H("button", {
						key: 0,
						type: "button",
						disabled: I(J).training.running || I(Y)("oww-retry"),
						onClick: m[20] ||= (e) => I(bc)(String(C.value.key || ""))
					}, k(I(Y)("oww-retry") ? "Starting OWW…" : "Retry OWW only"), 9, Zd)) : K("", !0),
					w.value ? (V(), H("button", {
						key: 1,
						type: "button",
						disabled: I(J).training.running || I(Y)("nww-revalidate"),
						onClick: m[21] ||= (e) => I(xc)(String(w.value.id || "").replace("nano_", ""), String(I(J).session.safe_word || ""))
					}, k(I(Y)("nww-revalidate") ? "Checking NWW…" : `Revalidate ${String(w.value.label || "NWW")} candidate`), 9, Qd)) : K("", !0)
				])])
			])) : (V(), H("section", $d, [
				U("h3", null, "Ready to train “" + k(I(J).phrase) + "”", 1),
				U("dl", ef, [
					m[113] ||= U("dt", null, "Language", -1),
					U("dd", null, k(I(J).languages.find((e) => e.code === I(J).language)?.label || I(J).language), 1),
					m[114] ||= U("dt", null, "Voice generation", -1),
					U("dd", null, k(I(Is)), 1),
					m[115] ||= U("dt", null, "English accent", -1),
					U("dd", null, k(I(J).englishAccents.find((e) => e.code === I(J).englishAccent)?.label || I(J).englishAccent), 1),
					m[116] ||= U("dt", null, "Recordings", -1),
					U("dd", null, k(I(Ns)) + " personal · " + k(I(Ps)) + " reviewed negatives · generated voices included", 1),
					m[117] ||= U("dt", null, "Models", -1),
					U("dd", null, k(l.value.join(", ")), 1),
					m[118] ||= U("dt", null, "Phrases to reject", -1),
					U("dd", null, k(de(I(J).generatedNegativePhrases)) + " configured", 1),
					m[119] ||= U("dt", null, "Your results", -1),
					m[120] ||= U("dd", null, "Saved validation report, version history, bundle JSON, and ESPHome JSON when MicroWakeWord is included.", -1)
				]),
				m[122] ||= U("p", { class: "workspace-note" }, "Earlier saved versions and their JSON links stay available throughout training.", -1),
				U("div", tf, [U("button", {
					type: "button",
					onClick: m[22] ||= (e) => I(J).wizardStep = 2
				}, "Back to models"), U("button", {
					type: "button",
					class: "primary",
					disabled: I(J).training.running || I(Y)("training-start") || !I(J).session.safe_word,
					onClick: m[23] ||= (...e) => I(vc) && I(vc)(...e)
				}, [W($, { name: "activity" }), m[121] ||= G("Start training", -1)], 8, nf)])
			]))], 64)) : I(J).activeView === "auto" ? (V(), H(B, { key: 6 }, [
				U("section", rf, [m[123] ||= U("div", null, [
					U("span", { class: "eyebrow" }, "False-positive loop"),
					U("h2", null, "Auto Training"),
					U("p", null, "Transcribe captures, sort negatives, recover close misses, retrain on schedule, and publish through Tater.")
				], -1), U("span", { class: O(["pill hero-pill", _.value.tone]) }, k(_.value.text), 3)]),
				U("section", af, [
					m[133] ||= U("header", { class: "panel-head" }, [U("div", { class: "number" }, "1"), U("div", null, [U("h3", null, "Review rules"), U("p", null, "Conservative local STT keeps uncertain clips in the manual inbox.")])], -1),
					U("div", of, [
						U("label", null, [R(U("input", {
							"onUpdate:modelValue": m[24] ||= (e) => I(J).autoForm.enabled = e,
							type: "checkbox"
						}, null, 512), [[$o, I(J).autoForm.enabled]]), m[124] ||= U("span", null, [U("strong", null, "Enable Auto Training"), U("small", null, "Queue eligible wake triggers for local transcription.")], -1)]),
						U("label", null, [R(U("input", {
							"onUpdate:modelValue": m[25] ||= (e) => I(J).autoForm.delete_confirmed_wakes = e,
							type: "checkbox"
						}, null, 512), [[$o, I(J).autoForm.delete_confirmed_wakes]]), m[125] ||= U("span", null, [U("strong", null, "Delete confirmed good wakes"), U("small", null, "Remove normal triggers when STT confirms the phrase.")], -1)]),
						U("label", null, [R(U("input", {
							"onUpdate:modelValue": m[26] ||= (e) => I(J).autoForm.promote_close_misses = e,
							type: "checkbox"
						}, null, 512), [[$o, I(J).autoForm.promote_close_misses]]), m[126] ||= U("span", null, [U("strong", null, "Promote confirmed close misses"), U("small", null, "Move verified close misses into positive samples.")], -1)])
					]),
					U("div", sf, [
						U("label", cf, [m[127] ||= U("span", null, "Wake phrase", -1), R(U("input", {
							"onUpdate:modelValue": m[27] ||= (e) => I(J).autoForm.wake_phrase = e,
							type: "text"
						}, null, 512), [[Qo, I(J).autoForm.wake_phrase]])]),
						U("label", lf, [m[128] ||= U("span", null, "STT language", -1), R(U("input", {
							"onUpdate:modelValue": m[28] ||= (e) => I(J).autoForm.language = e,
							type: "text"
						}, null, 512), [[Qo, I(J).autoForm.language]])]),
						String(I(J).autoForm.language).toLowerCase().startsWith("en") ? (V(), H("label", uf, [
							m[129] ||= U("span", null, "English accent emphasis", -1),
							R(U("select", { "onUpdate:modelValue": m[29] ||= (e) => I(J).autoForm.english_accent = e }, [(V(!0), H(B, null, z(I(J).englishAccents, (e) => (V(), H("option", {
								key: e.code,
								value: e.code
							}, k(e.label), 9, df))), 128))], 512), [[ts, I(J).autoForm.english_accent]]),
							m[130] ||= U("small", null, "Used when Auto Training needs to regenerate English TTS.", -1)
						])) : K("", !0),
						U("label", ff, [
							m[131] ||= U("span", null, "STT engine", -1),
							R(U("select", { "onUpdate:modelValue": m[30] ||= (e) => I(J).autoForm.stt_engine = e }, [(V(!0), H(B, null, z(I(Bs), (e) => (V(), H("option", {
								key: e.id || e.value,
								value: e.id || e.value
							}, k(e.label || e.name || e.id), 9, pf))), 128))], 512), [[ts, I(J).autoForm.stt_engine]]),
							U("small", null, k(I(Bs).find((e) => (e.id || e.value) === I(J).autoForm.stt_engine)?.description || "Runs locally on this trainer."), 1)
						]),
						U("label", mf, [m[132] ||= U("span", null, "Minimum transcript characters", -1), R(U("input", {
							"onUpdate:modelValue": m[31] ||= (e) => I(J).autoForm.minimum_transcript_chars = e,
							min: "1",
							max: "100",
							type: "number"
						}, null, 512), [[
							Qo,
							I(J).autoForm.minimum_transcript_chars,
							void 0,
							{ number: !0 }
						]])])
					])
				]),
				U("section", hf, [
					m[153] ||= U("header", { class: "panel-head" }, [U("div", { class: "number" }, "2"), U("div", null, [U("h3", null, "Training schedule"), U("p", null, "A run starts only after enough newly reviewed negatives accumulate.")])], -1),
					U("div", gf, [U("label", _f, [m[135] ||= U("span", null, "Run training", -1), R(U("select", { "onUpdate:modelValue": m[32] ||= (e) => I(J).autoForm.schedule_hours = e }, [...m[134] ||= [
						U("option", { value: 0 }, "Manually only", -1),
						U("option", { value: 6 }, "Every 6 hours", -1),
						U("option", { value: 12 }, "Every 12 hours", -1),
						U("option", { value: 24 }, "Every day", -1),
						U("option", { value: 48 }, "Every 2 days", -1),
						U("option", { value: 168 }, "Every week", -1)
					]], 512), [[
						ts,
						I(J).autoForm.schedule_hours,
						void 0,
						{ number: !0 }
					]])]), U("label", vf, [m[136] ||= U("span", null, "Minimum new negatives", -1), R(U("input", {
						"onUpdate:modelValue": m[33] ||= (e) => I(J).autoForm.minimum_new_negatives = e,
						min: "1",
						max: "10000",
						type: "number"
					}, null, 512), [[
						Qo,
						I(J).autoForm.minimum_new_negatives,
						void 0,
						{ number: !0 }
					]])])]),
					U("div", yf, [
						U("div", bf, [
							U("button", {
								type: "button",
								class: O(["model-choice", { active: I(J).autoForm.train_microwakeword }]),
								"aria-pressed": I(J).autoForm.train_microwakeword,
								onClick: m[34] ||= (e) => I(J).autoForm.train_microwakeword = !I(J).autoForm.train_microwakeword
							}, [...m[137] ||= [
								U("i", null, "MWW", -1),
								U("span", null, [U("strong", null, "MicroWakeWord"), U("small", null, "Refresh the ESP32 and ESPHome model.")], -1),
								U("b", null, "✓", -1)
							]], 10, xf),
							U("button", {
								type: "button",
								class: O(["model-choice", { active: I(J).autoForm.train_openwakeword }]),
								"aria-pressed": I(J).autoForm.train_openwakeword,
								onClick: m[35] ||= (e) => I(J).autoForm.train_openwakeword = !I(J).autoForm.train_openwakeword
							}, [...m[138] ||= [
								U("i", null, "OWW", -1),
								U("span", null, [U("strong", null, "openWakeWord"), U("small", null, "Refresh the openWakeWord ONNX model.")], -1),
								U("b", null, "✓", -1)
							]], 10, Sf),
							U("button", {
								type: "button",
								class: O(["model-choice", { active: T.value }]),
								"aria-pressed": T.value,
								onClick: ue
							}, [...m[139] ||= [
								U("i", null, "NWW", -1),
								U("span", null, [U("strong", null, "NanoWakeWord"), U("small", null, "Refresh selected CNN and GRU models.")], -1),
								U("b", null, "✓", -1)
							]], 10, Cf)
						]),
						T.value ? (V(), H("div", wf, [U("div", Tf, [m[140] ||= U("div", null, [U("strong", null, "NanoWakeWord architectures"), U("small", null, "Select either architecture or retrain both.")], -1), U("span", null, k(I(J).autoForm.train_nanowakeword_cnn && I(J).autoForm.train_nanowakeword_gru ? "2 selected" : "1 selected"), 1)]), U("div", Ef, [U("button", {
							type: "button",
							class: O(["architecture-choice", { active: I(J).autoForm.train_nanowakeword_cnn }]),
							"aria-pressed": I(J).autoForm.train_nanowakeword_cnn,
							onClick: m[36] ||= (e) => I(J).autoForm.train_nanowakeword_cnn = !I(J).autoForm.train_nanowakeword_cnn
						}, [...m[141] ||= [
							U("i", null, "CNN", -1),
							U("span", null, [U("strong", null, "CNN"), U("small", null, "Recommended default with stronger recall.")], -1),
							U("b", null, "✓", -1)
						]], 10, Df), U("button", {
							type: "button",
							class: O(["architecture-choice", { active: I(J).autoForm.train_nanowakeword_gru }]),
							"aria-pressed": I(J).autoForm.train_nanowakeword_gru,
							onClick: m[37] ||= (e) => I(J).autoForm.train_nanowakeword_gru = !I(J).autoForm.train_nanowakeword_gru
						}, [...m[142] ||= [
							U("i", null, "GRU", -1),
							U("span", null, [U("strong", null, "GRU"), U("small", null, "Recurrent model tuned for fewer false triggers.")], -1),
							U("b", null, "✓", -1)
						]], 10, Of)])])) : K("", !0),
						E.value.length ? (V(), H("section", kf, [
							U("header", Af, [
								m[143] ||= U("div", { class: "hard-negative-icon" }, "−", -1),
								m[144] ||= U("div", null, [
									U("span", { class: "eyebrow" }, "Optional precision tuning"),
									U("strong", null, "Hard negative training"),
									U("small", null, "Keep near-miss protection consistent whenever Auto Training refreshes a model.")
								], -1),
								U("b", null, k(de(I(J).autoForm.generated_negative_phrases) ? `${de(I(J).autoForm.generated_negative_phrases)} phrase${de(I(J).autoForm.generated_negative_phrases) === 1 ? "" : "s"}` : "Optional"), 1)
							]),
							U("div", jf, [m[145] ||= U("span", null, "Applied to", -1), (V(), H(B, null, z([
								"MWW",
								"OWW",
								"NWW"
							], (e) => U("i", {
								key: e,
								class: O({ active: E.value.includes(e) })
							}, k(e), 3)), 64))]),
							U("label", Mf, [
								m[146] ||= U("span", null, "Phrases to reject", -1),
								R(U("textarea", {
									"onUpdate:modelValue": m[38] ||= (e) => I(J).autoForm.generated_negative_phrases = e,
									rows: "4",
									placeholder: "hey later\nhey waiter\nokay tater"
								}, null, 512), [[Qo, I(J).autoForm.generated_negative_phrases]]),
								m[147] ||= U("small", null, "One phrase per line. Saved with the schedule and synthesized for every selected model.", -1)
							]),
							m[148] ||= U("footer", null, [U("span", null, "Reviewed false wakes still receive the strongest model weight."), U("strong", null, "The actual wake phrase is ignored automatically.")], -1)
						])) : K("", !0)
					]),
					!I(J).training.running && I(J).training.summary?.models?.length ? (V(), H("section", Nf, [U("header", null, [
						U("div", { class: O(["summary-mark", I(J).training.summary.tone]) }, k(I(J).training.summary.tone === "success" ? "✓" : I(J).training.summary.tone === "error" ? "×" : "!"), 3),
						U("div", null, [
							m[149] ||= U("span", { class: "eyebrow" }, "Last completed run", -1),
							U("h4", null, k(I(J).training.summary.wake_word), 1),
							U("p", null, [G(k(I(J).training.summary.message), 1), I(J).training.summary.completed_at ? (V(), H(B, { key: 0 }, [G(" · " + k(I(Oc)(I(J).training.summary.completed_at)), 1)], 64)) : K("", !0)])
						]),
						U("span", { class: O(["pill", I(J).training.summary.tone]) }, k(I(J).training.summary.overall), 3)
					]), U("div", Pf, [(V(!0), H(B, null, z(I(J).training.summary.models, (e) => (V(), H("article", {
						key: `auto-${e.id}`,
						class: O(e.tone)
					}, [
						U("header", null, [U("div", null, [U("span", null, k(e.id.replaceAll("_", " ")), 1), U("strong", null, k(e.label), 1)]), U("b", null, k(e.rating), 1)]),
						U("p", null, k(e.message), 1),
						e.metrics?.length ? (V(), H("div", Ff, [(V(!0), H(B, null, z(e.metrics, (e) => (V(), H("span", { key: `${e.label}-${e.value}` }, [U("small", null, k(e.label), 1), U("strong", null, k(e.value), 1)]))), 128))])) : K("", !0)
					], 2))), 128))])])) : K("", !0),
					U("div", If, [
						U("article", null, [m[150] ||= U("span", null, "Pending negatives", -1), U("strong", null, k(Number(p.value.pending_negative_count || 0)), 1)]),
						U("article", null, [m[151] ||= U("span", null, "Next check", -1), U("strong", Lf, k(p.value.next_run_at ? I(Oc)(p.value.next_run_at) : "Manual"), 1)]),
						U("article", null, [m[152] ||= U("span", null, "Last training", -1), U("strong", Rf, k(p.value.last_train_finished_at ? I(Oc)(p.value.last_train_finished_at) : "Never"), 1)])
					])
				]),
				U("section", zf, [
					m[157] ||= U("header", { class: "panel-head" }, [U("div", { class: "number" }, "3"), U("div", null, [
						U("span", { class: "eyebrow" }, "Optional · Model updates"),
						U("h3", null, "Auto-training link"),
						U("p", null, "Automatically apply newly trained wake models. Satellite audio uploads do not need this link.")
					])], -1),
					U("div", Bf, [U("label", Vf, [
						m[154] ||= U("span", null, "Trainer public URL", -1),
						R(U("input", {
							"onUpdate:modelValue": m[39] ||= (e) => I(J).autoForm.advertised_base_url = e,
							type: "text",
							placeholder: "Auto-detect LAN address"
						}, null, 512), [[Qo, I(J).autoForm.advertised_base_url]]),
						U("small", null, k(I(J).autoForm.advertised_base_url ? `Configured: ${I(J).autoForm.advertised_base_url}` : `Detected: ${I(J).auto.advertised_base_url || "unavailable"}`), 1)
					]), U("label", Hf, [m[155] ||= U("span", null, "Tater URL", -1), R(U("input", {
						"onUpdate:modelValue": m[40] ||= (e) => I(J).autoForm.tater_url = e,
						type: "text"
					}, null, 512), [[Qo, I(J).autoForm.tater_url]])])]),
					U("div", Uf, [
						U("span", { class: O(["pill", I(zs) ? "success" : ""]) }, k(I(zs) ? `Linked${I(J).auto.trainer_link?.tater_name ? ` · ${I(J).auto.trainer_link.tater_name}` : ""}` : "Not linked · optional"), 3),
						U("button", {
							type: "button",
							class: "button primary",
							disabled: I(Y)("auto"),
							onClick: pe
						}, k(I(zs) ? "Relink Tater" : "Link Tater"), 9, Wf),
						I(zs) ? (V(), H("button", {
							key: 0,
							type: "button",
							class: "button danger",
							disabled: I(Y)("auto"),
							onClick: m[41] ||= (...e) => I(lc) && I(lc)(...e)
						}, "Unlink", 8, Gf)) : K("", !0)
					]),
					U("div", Kf, [U("label", null, [R(U("input", {
						"onUpdate:modelValue": m[42] ||= (e) => I(J).autoForm.notify_satellites = e,
						type: "checkbox"
					}, null, 512), [[$o, I(J).autoForm.notify_satellites]]), m[156] ||= U("span", null, [U("strong", null, "Activate after successful training"), U("small", null, "Tater applies the new word globally.")], -1)])])
				]),
				U("section", qf, [U("div", Jf, [
					U("button", {
						type: "button",
						class: "button primary",
						disabled: I(Y)("auto"),
						onClick: m[43] ||= (...e) => I(oc) && I(oc)(...e)
					}, "Save Auto Training", 8, Yf),
					U("button", {
						type: "button",
						disabled: I(Y)("auto"),
						onClick: m[44] ||= (e) => I(sc)("review_now")
					}, "Review inbox now", 8, Xf),
					U("button", {
						type: "button",
						disabled: I(Y)("auto") || I(J).training.running,
						onClick: m[45] ||= (e) => I(sc)("train_now")
					}, "Train now", 8, Zf),
					U("button", {
						type: "button",
						disabled: I(Y)("auto") || !I(zs),
						onClick: m[46] ||= (e) => I(sc)("notify_now")
					}, "Publish current word", 8, Qf)
				]), U("p", $f, k(h.value), 1)])
			], 64)) : I(J).activeView === "captured" ? (V(), H(B, { key: 7 }, [
				U("section", ep, [m[158] ||= U("div", null, [
					U("span", { class: "eyebrow" }, "Capture review"),
					U("h2", null, "Captured Audio"),
					U("p", null, "Listen to clips from your satellites and turn every real-world event into a better model.")
				], -1), U("span", { class: O(["pill hero-pill", I(J).captured.captured_count ? "warning" : ""]) }, k(I(J).captured.captured_count ? `${I(J).captured.captured_count} waiting` : "Inbox idle"), 3)]),
				U("section", tp, [U("header", np, [
					m[159] ||= U("div", { class: "number" }, "1", -1),
					m[160] ||= U("div", null, [U("h3", null, "Review queue"), U("p", null, "Approve good phrases, keep false positives as negatives, or discard noise.")], -1),
					U("button", {
						type: "button",
						disabled: I(Y)("captured"),
						onClick: m[47] ||= (e) => I(Xs)()
					}, k(I(Y)("captured") ? "Refreshing…" : "Refresh inbox"), 9, rp)
				]), U("div", ip, [
					U("article", null, [m[161] ||= U("span", null, "Inbox", -1), U("strong", null, k(I(J).captured.captured_count), 1)]),
					U("article", null, [m[162] ||= U("span", null, "All-word negatives", -1), U("strong", null, k(I(J).captured.negative_count), 1)]),
					U("article", null, [m[163] ||= U("span", null, "All-word positives", -1), U("strong", null, k(I(J).captured.personal_count), 1)])
				])]),
				U("section", ap, [m[168] ||= U("header", { class: "panel-head" }, [U("div", { class: "number" }, "2"), U("div", null, [U("h3", null, "Listen + sort"), U("p", null, "Metadata remains visible so borderline detections are easy to understand.")])], -1), I(J).captured.items?.length ? (V(), H("div", sp, [(V(!0), H(B, null, z(I(J).captured.items, (e) => (V(), H("article", {
					key: e.capture_id || e.saved_as,
					class: "audio-card"
				}, [
					U("header", null, [U("div", null, [U("strong", null, k(e.original_name || e.saved_as), 1), U("small", null, k(I(Oc)(e.captured_at || e.received_at)) + " " + k(e.message || ""), 1)]), U("span", { class: O(["pill", I(jc)(e).tone]) }, k(I(jc)(e).label), 3)]),
					ge(e).length ? (V(), H("div", cp, [(V(!0), H(B, null, z(ge(e), (e) => (V(), H("span", { key: e }, k(e), 1))), 128))])) : K("", !0),
					U("div", { class: O(["capture-assignment", { unassigned: !e.review_wake_word }]) }, [U("label", null, [m[165] ||= U("span", null, "Train this clip for", -1), R(U("select", { "onUpdate:modelValue": (t) => e.review_wake_word = t }, [m[164] ||= U("option", { value: "" }, "Choose a saved wake word…", -1), (V(!0), H(B, null, z(I(J).wakeWordLibrary, (e) => (V(), H("option", {
						key: e.safe_word,
						value: e.safe_word
					}, k(e.phrase) + " · " + k(e.safe_word), 9, up))), 128))], 8, lp), [[ts, e.review_wake_word]])]), U("span", { class: O(["pill", e.review_wake_word ? "success" : "warning"]) }, k(e.review_wake_word ? `Routed to ${e.review_wake_word}` : "Unassigned"), 3)], 2),
					e.transcript ? (V(), H("div", dp, [m[166] ||= U("b", null, "STT", -1), G(" " + k(e.transcript), 1)])) : K("", !0),
					e.auto_review_guided_transcript ? (V(), H("div", fp, [m[167] ||= U("b", null, "Guided wake check", -1), G(" " + k(e.auto_review_guided_transcript), 1)])) : K("", !0),
					U("audio", {
						controls: "",
						preload: "none",
						src: I(Mc)(e, "captured")
					}, null, 8, pp),
					U("footer", null, [U("span", null, k(e.saved_as) + " · " + k(I(Ac)(e.final_format)), 1), U("div", null, [
						U("button", {
							type: "button",
							disabled: I(Y)("review") || !e.review_wake_word,
							onClick: (t) => I(ec)(e, "approve_personal")
						}, "Add positive", 8, mp),
						U("button", {
							type: "button",
							disabled: I(Y)("review") || !e.review_wake_word,
							onClick: (t) => I(ec)(e, "mark_negative")
						}, "Mark negative", 8, hp),
						U("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Y)("review"),
							onClick: (t) => I(ec)(e, "discard")
						}, "Discard", 8, gp)
					])])
				]))), 128))])) : (V(), H("div", op, "No captured audio yet. Clips sent by satellites will appear here."))])
			], 64)) : I(J).activeView === "samples" || I(J).activeView === "trainer" && I(J).wizardStep === 1 ? (V(), H(B, { key: 8 }, [
				I(J).activeView === "samples" ? (V(), H("section", _p, [m[169] ||= U("div", null, [
					U("span", { class: "eyebrow" }, "Sample library"),
					U("h2", null, "Current Training Samples"),
					U("p", null, "Audit positives and negatives, trim recordings precisely, and import seed audio.")
				], -1), U("span", vp, k(I(Ns) + I(Ps)) + " total", 1)])) : K("", !0),
				U("section", yp, [
					U("header", bp, [
						m[172] ||= U("div", { class: "number" }, "1", -1),
						m[173] ||= U("div", null, [U("h3", null, "Saved samples"), U("p", null, "Personal clips are positives. Negative clips are false wakes and hard negatives.")], -1),
						U("div", xp, [U("button", {
							type: "button",
							class: O({ active: I(J).sampleBucket === "personal" }),
							onClick: m[48] ||= (e) => D("personal")
						}, [m[170] ||= G("Personal ", -1), U("b", null, k(I(Ns)), 1)], 2), U("button", {
							type: "button",
							class: O({ active: I(J).sampleBucket === "negative" }),
							onClick: m[49] ||= (e) => D("negative")
						}, [m[171] ||= G("Negative ", -1), U("b", null, k(I(Ps)), 1)], 2)])
					]),
					U("div", Sp, [
						U("button", {
							type: "button",
							disabled: I(Y)("samples"),
							onClick: m[50] ||= (e) => I(Ys)()
						}, "Refresh", 8, Cp),
						U("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Y)("review") || I(Ns) === 0,
							onClick: m[51] ||= (e) => I(rc)("personal")
						}, "Clear positives", 8, wp),
						U("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Y)("review") || I(Ps) === 0,
							onClick: m[52] ||= (e) => I(rc)("negative")
						}, "Clear negatives", 8, Tp)
					]),
					I(Rs).length ? (V(), H("div", Dp, [(V(!0), H(B, null, z(d.value, (e) => (V(), H("article", {
						key: e.saved_as,
						class: "audio-card"
					}, [
						U("header", null, [U("div", null, [U("strong", null, k(e.saved_as), 1), U("small", null, k(_e(e)), 1)]), U("div", Op, [e.trimmed ? (V(), H("span", kp, "Trimmed")) : K("", !0), U("span", { class: O(["pill", I(J).sampleBucket === "personal" ? "success" : "error"]) }, k(I(J).sampleBucket === "personal" ? "Positive" : "Negative"), 3)])]),
						e.transcript ? (V(), H("div", Ap, [m[174] ||= U("b", null, "STT", -1), G(" " + k(e.transcript), 1)])) : K("", !0),
						e.auto_review_guided_transcript ? (V(), H("div", jp, [m[175] ||= U("b", null, "Guided wake check", -1), G(" " + k(e.auto_review_guided_transcript), 1)])) : K("", !0),
						U("audio", {
							controls: "",
							preload: "none",
							src: I(Mc)(e, I(J).sampleBucket)
						}, null, 8, Mp),
						U("footer", null, [U("span", null, k(I(Ac)(e.final_format)), 1), U("div", null, [
							U("button", {
								type: "button",
								onClick: (t) => ce(e, I(J).sampleBucket)
							}, "Trim", 8, Np),
							e.trimmed ? (V(), H("button", {
								key: 0,
								type: "button",
								onClick: (t) => I(nc)(e, I(J).sampleBucket)
							}, "Revert", 8, Pp)) : K("", !0),
							U("button", {
								type: "button",
								class: "button danger ghost",
								disabled: I(Y)("review"),
								onClick: (t) => I(tc)(e, I(J).sampleBucket)
							}, "Remove", 8, Fp)
						])])
					]))), 128))])) : (V(), H("div", Ep, "No " + k(I(J).sampleBucket) + " samples saved yet.", 1)),
					f.value > 1 ? (V(), H("div", Ip, [
						U("button", {
							type: "button",
							disabled: I(J).samplePage[I(J).sampleBucket] === 0,
							onClick: m[53] ||= (e) => I(J).samplePage[I(J).sampleBucket]--
						}, "Previous", 8, Lp),
						U("span", null, "Page " + k(I(J).samplePage[I(J).sampleBucket] + 1) + " of " + k(f.value), 1),
						U("button", {
							type: "button",
							disabled: I(J).samplePage[I(J).sampleBucket] >= f.value - 1,
							onClick: m[54] ||= (e) => I(J).samplePage[I(J).sampleBucket]++
						}, "Next", 8, Rp)
					])) : K("", !0)
				]),
				U("section", zp, [
					m[177] ||= U("header", { class: "panel-head" }, [U("div", { class: "number" }, "2"), U("div", null, [U("h3", null, "Manual sample import"), U("p", null, "Optional seed recordings are normalized to the trainer’s required WAV format.")])], -1),
					U("label", Bp, [
						U("input", {
							ref_key: "uploadInput",
							ref: t,
							type: "file",
							multiple: "",
							accept: "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.webm,.opus",
							onChange: m[55] ||= (...e) => I(Zs) && I(Zs)(...e)
						}, null, 544),
						m[176] ||= U("span", null, [U("strong", null, "Choose one or many audio files"), U("small", null, "WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, and WEBM")], -1),
						U("b", null, k(I(J).selectedFiles.length ? `${I(J).selectedFiles.length} selected` : "Browse"), 1)
					]),
					U("button", {
						type: "button",
						class: "button primary",
						disabled: !I(J).session.safe_word || !I(J).selectedFiles.length || I(Y)("upload"),
						onClick: m[56] ||= (e) => I($s)(t.value)
					}, k(I(Y)("upload") ? "Uploading…" : "Upload selected samples"), 9, Vp),
					U("div", Hp, [
						U("div", null, [U("strong", null, k(I(J).uploadLabel), 1), U("span", null, k(I(J).uploadProgress) + "%", 1)]),
						U("div", Up, [U("i", { style: fe({ width: `${I(J).uploadProgress}%` }) }, null, 4)]),
						U("small", null, k(I(J).uploadDetail), 1)
					])
				]),
				I(J).activeView === "trainer" ? (V(), H("div", Wp, [U("button", {
					type: "button",
					onClick: m[57] ||= (e) => I(J).wizardStep = 0
				}, "Back to phrase"), U("button", {
					type: "button",
					class: "primary",
					onClick: m[58] ||= (e) => I(J).wizardStep = 2
				}, [m[178] ||= G("Continue to models", -1), W($, { name: "next" })])])) : K("", !0)
			], 64)) : I(J).activeView === "data" ? (V(), H(B, { key: 9 }, [
				U("section", Gp, [m[179] ||= U("div", null, [
					U("span", { class: "eyebrow" }, "Local storage"),
					U("h2", null, "Data Management"),
					U("p", null, "See exactly what the trainer has downloaded, generated, recorded, and produced.")
				], -1), U("span", Kp, k(I(kc)(I(J).managedData.total_size_bytes)) + " total", 1)]),
				(V(), H("section", qp, [
					U("header", Jp, [
						m[180] ||= U("div", { class: "number storage-icon" }, "↗", -1),
						m[181] ||= U("div", null, [U("h3", null, "Data folder"), U("p", null, "Put recordings, datasets, model downloads, caches, training environments, and results on any Mac-compatible drive.")], -1),
						I(Ds) ? (V(), H("button", {
							key: 0,
							type: "button",
							disabled: I(J).training.running || I(Y)("data") || I(Y)("data-delete"),
							onClick: m[59] ||= (...e) => I(hc) && I(hc)(...e)
						}, "Change Folder…", 8, Yp)) : K("", !0)
					]),
					U("div", Xp, [U("div", null, [m[182] ||= U("span", null, "Current location", -1), U("code", null, k(I(J).managedData.data_directory || "Loading…"), 1)]), I(Ds) ? (V(), H("span", Zp, "Mac app")) : (V(), H("span", Qp, "Open the Mac app to change"))]),
					m[183] ||= U("p", { class: "storage-hint" }, "Changing this folder safely stops the local trainer, copies and verifies the data, then restarts it. Keep an external drive connected whenever you use the trainer.", -1),
					I(J).training.running ? (V(), H("p", $p, "Stop the active training session before changing the data folder.")) : K("", !0)
				])),
				U("section", em, [
					U("header", tm, [
						m[184] ||= U("div", { class: "number" }, "i", -1),
						m[185] ||= U("div", null, [U("h3", null, "Trainer storage"), U("p", null, "Deleting an item is permanent. Required downloads and generated caches will be rebuilt the next time training needs them.")], -1),
						U("button", {
							type: "button",
							disabled: I(Y)("data") || I(Y)("data-delete"),
							onClick: m[60] ||= (e) => I(mc)()
						}, k(I(Y)("data") ? "Scanning…" : "Refresh sizes"), 9, nm)
					]),
					U("div", rm, [
						U("article", null, [m[186] ||= U("span", null, "Space used", -1), U("strong", im, k(I(kc)(I(J).managedData.total_size_bytes)), 1)]),
						U("article", null, [m[187] ||= U("span", null, "Files", -1), U("strong", null, k(Number(I(J).managedData.total_file_count || 0).toLocaleString()), 1)]),
						U("article", null, [m[188] ||= U("span", null, "Individual items", -1), U("strong", null, k(I(J).managedData.items.length), 1)])
					]),
					I(J).training.running ? (V(), H("p", am, "Stop the active training session before deleting data.")) : K("", !0)
				]),
				(V(!0), H(B, null, z(te.value, (e, t) => (V(), H("section", {
					key: e.name,
					class: "panel data-panel"
				}, [U("header", om, [U("div", sm, k(t + 1), 1), U("div", null, [U("h3", null, k(e.name), 1), U("p", null, k(e.items.length) + " separately managed item" + k(e.items.length === 1 ? "" : "s"), 1)])]), U("div", cm, [(V(!0), H(B, null, z(e.items, (e) => (V(), H("article", {
					key: e.id,
					class: O(["data-row", { empty: !e.file_count }])
				}, [
					U("div", lm, [
						U("div", um, [U("strong", null, k(e.label), 1), U("code", null, k(e.location), 1)]),
						U("small", null, k(e.description), 1),
						e.rebuild_note ? (V(), H("span", dm, k(e.rebuild_note), 1)) : K("", !0)
					]),
					U("div", fm, [U("strong", null, k(I(kc)(e.size_bytes)), 1), U("span", null, k(Number(e.file_count || 0).toLocaleString()) + " file" + k(e.file_count === 1 ? "" : "s"), 1)]),
					U("button", {
						type: "button",
						class: "button danger ghost",
						disabled: !e.file_count || I(J).training.running || I(Y)("data") || I(Y)("data-delete"),
						onClick: (t) => I(gc)(e)
					}, k(I(Y)("data-delete") ? "Please wait…" : "Delete"), 9, pm)
				], 2))), 128))])]))), 128)),
				!I(Y)("data") && !I(J).managedData.items.length ? (V(), H("section", mm, "No managed trainer data was found.")) : K("", !0)
			], 64)) : K("", !0)], 64)) : (V(), H("div", Fu, [...m[70] ||= [U("span", { class: "spinner" }, null, -1), U("strong", null, "Connecting to the trainer…", -1)]]))]),
			I(J).initialized ? (V(), H("footer", hm, [U("span", gm, [
				W($, { name: "link" }),
				G(k(I(zs) ? "Tater linked" : "Local trainer"), 1),
				U("button", {
					type: "button",
					class: "text-button",
					onClick: pe
				}, k(I(zs) ? "Connection" : "Link Tater"), 1)
			]), m[189] ||= U("span", null, "Recordings, results, and every saved version stay with the word.", -1)])) : K("", !0),
			(V(), aa(Jn, { to: "body" }, [I(J).taterLinkOpen ? (V(), H("div", {
				key: 0,
				class: "modal-backdrop",
				onClick: m[64] ||= ss((e) => I(J).taterLinkOpen = !1, ["self"])
			}, [U("section", _m, [
				U("header", vm, [U("div", null, [
					m[190] ||= U("span", { class: "eyebrow" }, "Optional · Auto-training link", -1),
					U("h2", null, k(o.value ? "Tater linked" : "Link Tater"), 1),
					U("p", null, k(o.value ? "This trainer can securely publish wake-word updates." : "Enter the short-lived code shown in Tater Voice Settings."), 1)
				]), U("button", {
					type: "button",
					onClick: m[61] ||= (e) => I(J).taterLinkOpen = !1
				}, "Close")]),
				m[196] ||= U("p", { class: "workspace-note" }, "For model updates only. To send audio, set this trainer’s URL in Tater or the HA bridge. No link required.", -1),
				o.value ? (V(), H("div", ym, [
					m[191] ||= U("i", null, "✓", -1),
					U("strong", null, "Successfully linked" + k(I(J).auto.trainer_link?.tater_name ? ` to ${I(J).auto.trainer_link.tater_name}` : ""), 1),
					m[192] ||= U("span", null, "The private link key is stored locally and is never displayed.", -1)
				])) : (V(), H("div", bm, [
					U("label", xm, [m[193] ||= U("span", null, "Tater address", -1), R(U("input", {
						"onUpdate:modelValue": m[62] ||= (e) => i.value = e,
						type: "text"
					}, null, 512), [[Qo, i.value]])]),
					U("label", Sm, [m[194] ||= U("span", null, "Tater pairing code", -1), R(U("input", {
						id: "pairing-code",
						"onUpdate:modelValue": m[63] ||= (e) => a.value = e,
						class: "pairing-code",
						maxlength: "9",
						placeholder: "ABCD-EFGH",
						autocomplete: "off",
						onInput: me
					}, null, 544), [[Qo, a.value]])]),
					m[195] ||= U("small", null, "In Tater or the HA bridge, open the Auto-training link section.", -1),
					U("button", {
						type: "button",
						class: "button primary",
						disabled: I(Y)("link"),
						onClick: he
					}, k(I(Y)("link") ? "Linking securely…" : "Link Tater"), 9, Cm)
				]))
			])])) : K("", !0)])),
			W(Wc),
			W(to, { name: "toast" }, {
				default: Dn(() => [I(J).toast.message ? (V(), H("div", {
					key: 0,
					class: O(["toast", I(J).toast.tone]),
					role: "status"
				}, k(I(J).toast.message), 3)) : K("", !0)]),
				_: 1
			})
		]));
	}
}), Em = document.getElementById("trainer-app");
if (!Em) throw Error("Missing #trainer-app mount point");
ps(Tm).mount(Em);
//#endregion
