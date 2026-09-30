/* ============================================================================
   course site — the small amount of behavior the pages need.
   No framework, no build step: every page is a plain file that opens from
   disk as happily as it serves from GitHub Pages.
   ==========================================================================*/
(function () {
  'use strict';

  /* ── Tabs: same program, different representations ─────────────────────
     Used to put source / AST / TAC / MIPS side by side so the reader can
     flip between them without losing their place. */
  function initTabs(root) {
    var bar = root.querySelector('.tabbar');
    var panels = Array.prototype.slice.call(root.querySelectorAll('.panel'));
    var buttons = Array.prototype.slice.call(bar.querySelectorAll('button'));

    function show(i) {
      buttons.forEach(function (b, k) { b.setAttribute('aria-selected', k === i ? 'true' : 'false'); });
      panels.forEach(function (p, k) { p.hidden = (k !== i); });
    }
    buttons.forEach(function (b, i) {
      b.addEventListener('click', function () {
        show(i);
        // Deep-linkable tabs: keep the URL in step so a single tab can be
        // shared. replaceState rather than assigning location.hash, which
        // would scroll the page to the top of the widget on every click.
        if (b.dataset.tab) {
          history.replaceState(null, '', '#' + b.dataset.tab);
        }
      });
    });
    bar.addEventListener('keydown', function (e) {
      var i = buttons.indexOf(document.activeElement);
      if (i < 0) return;
      if (e.key === 'ArrowRight') { buttons[(i + 1) % buttons.length].focus(); show((i + 1) % buttons.length); }
      if (e.key === 'ArrowLeft')  { var j = (i - 1 + buttons.length) % buttons.length; buttons[j].focus(); show(j); }
    });
    // Arriving with a hash opens that tab.
    var want = decodeURIComponent((location.hash || '').replace(/^#/, ''));
    var start = 0;
    buttons.forEach(function (b, i) { if (b.dataset.tab === want) start = i; });
    show(start);
  }

  /* ── Stepper: walk one transformation one move at a time ────────────────
     A compiler phase is a sequence of small, individually obvious steps.
     Showing the finished output hides that; stepping through it does not. */
  function initStepper(root) {
    var steps = Array.prototype.slice.call(root.querySelectorAll('.step'));
    var prev = root.querySelector('[data-act="prev"]');
    var next = root.querySelector('[data-act="next"]');
    var reset = root.querySelector('[data-act="reset"]');
    var count = root.querySelector('.count');
    var i = 0;

    function show() {
      steps.forEach(function (s, k) { s.hidden = (k !== i); });
      prev.disabled = (i === 0);
      next.disabled = (i === steps.length - 1);
      count.textContent = 'step ' + (i + 1) + ' of ' + steps.length;
    }
    prev.addEventListener('click', function () { if (i > 0) { i--; show(); } });
    next.addEventListener('click', function () { if (i < steps.length - 1) { i++; show(); } });
    if (reset) reset.addEventListener('click', function () { i = 0; show(); });
    root.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight') next.click();
      if (e.key === 'ArrowLeft') prev.click();
    });
    show();
  }

  /* ── Quiz: check an idea before reading on ──────────────────────────────
     Deliberately ungraded and unrecorded. The feedback matters, the score
     does not. */
  function initQuiz(root) {
    var opts = Array.prototype.slice.call(root.querySelectorAll('.opt'));
    var fb = root.querySelector('.fb');
    opts.forEach(function (o) {
      o.addEventListener('click', function () {
        var right = o.dataset.correct === '1';
        o.classList.add(right ? 'right' : 'wrong');
        fb.textContent = o.dataset.feedback || (right ? 'Correct.' : 'Not quite.');
        fb.classList.add('shown');
        if (right) opts.forEach(function (x) { x.disabled = true; });
      });
    });
  }

  document.querySelectorAll('.tabs').forEach(initTabs);
  document.querySelectorAll('.stepper').forEach(initStepper);
  document.querySelectorAll('.quiz').forEach(initQuiz);
})();
