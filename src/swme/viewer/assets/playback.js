/*
 * Playback in the browser (swme/viewer/app.py, "playback").
 *
 * While playing, the clock runs here: every tick of the `player` interval
 * moves the time slider one stored step and draws that step's frame into the
 * visible graphs directly, without a server round trip per frame. The frames
 * themselves are built on the server by the same figure builders that draw
 * the paused view (`playback_frame`), fetched ahead of the cursor in batches
 * (`frame_batch`) and buffered here. If the next frame has not arrived yet,
 * the clock waits for it rather than skipping it.
 *
 * The server ignores the slider moves made while playing (its `_time`
 * callback). On the last step this sets `player.disabled` together with the
 * slider, so the server publishes that time and every view, including the
 * text panels, is redrawn exactly; pausing does the same through the play
 * button.
 *
 * A frame is {graph id: part}. A part is {figure} (drawn whole), or
 * {data: {trace index: {prop: value}}, layout: {title, annotations, shapes}}
 * (merged into what is on screen): the same content as the server's frame
 * patch (`_frame_patch`), so a played frame and a paused one are the same
 * picture.
 */
(function () {
    "use strict";

    var LOOKAHEAD = 48;        // frames requested ahead of the cursor
    var RETRY_MS = 5000;       // re-request a batch that never arrived

    var buffer = {context: null, frames: new Map(), pending: null, sentAt: 0, sequence: 0};

    function reset(context) {
        buffer.context = context;
        buffer.frames = new Map();
        buffer.pending = null;
    }

    function plot(id) {
        var host = document.getElementById(id);
        var node = host ? host.querySelector(".js-plotly-plot") : null;
        return node && node.data && node.layout ? node : null;
    }

    function drawPart(gd, part) {
        if (part.figure) {
            window.Plotly.react(gd, part.figure.data, part.figure.layout);
            return;
        }
        var changes = part.data || {};
        var data = gd.data.map(function (trace, index) {
            return changes[index] ? Object.assign({}, trace, changes[index]) : trace;
        });
        var layout = Object.assign({}, gd.layout);
        var update = part.layout || {};
        Object.keys(update).forEach(function (key) {
            layout[key] = key === "title"
                ? Object.assign({}, gd.layout.title, update.title)
                : update[key];
        });
        window.Plotly.react(gd, data, layout);
    }

    function draw(frame) {
        if (!window.Plotly) {
            return;
        }
        Object.keys(frame).forEach(function (id) {
            var gd = plot(id);
            if (gd) {
                drawPart(gd, frame[id]);
            }
        });
    }

    // The next batch to fetch, or no_update when enough is buffered or a
    // request is already on its way.
    function request(next, last) {
        var noUpdate = window.dash_clientside.no_update;
        if (buffer.pending !== null && Date.now() - buffer.sentAt < RETRY_MS) {
            return noUpdate;
        }
        var start = next;
        while (start <= last && buffer.frames.has(start)) {
            start += 1;
        }
        if (start > last || start - next >= LOOKAHEAD) {
            return noUpdate;
        }
        buffer.pending = start;
        buffer.sentAt = Date.now();
        buffer.sequence += 1;
        return {context: buffer.context, start: start, sequence: buffer.sequence,
                count: Math.min(LOOKAHEAD - (start - next), last - start + 1)};
    }

    function tick(_intervals, index, grid, disabled, context) {
        var noUpdate = window.dash_clientside.no_update;
        if (disabled || !grid || context === null || context === undefined) {
            return [noUpdate, noUpdate, noUpdate, noUpdate];
        }
        if (context !== buffer.context) {
            reset(context);
        }
        var last = grid.length - 1;
        var position = Math.min(Math.max(index || 0, 0), last);
        if (position >= last) {
            // Writing the slider, even unchanged, is what makes the server
            // publish this time now that the player is off.
            return [position, true, "▶", noUpdate];
        }
        var next = position + 1;
        var frame = buffer.frames.get(next);
        var fetch = request(next, last);
        if (frame === undefined) {
            return [noUpdate, noUpdate, noUpdate, fetch];
        }
        buffer.frames.forEach(function (_frame, key) {
            if (key < next) {
                buffer.frames.delete(key);
            }
        });
        draw(frame);
        var end = next >= last;
        return [next, end ? true : noUpdate, end ? "▶" : noUpdate, fetch];
    }

    function merge(batch) {
        if (batch && batch.context === buffer.context) {
            batch.frames.forEach(function (frame, offset) {
                buffer.frames.set(batch.start + offset, frame);
            });
        }
        if (batch && batch.start === buffer.pending) {
            buffer.pending = null;
        }
        return window.dash_clientside.no_update;
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        swmePlayback: {tick: tick, merge: merge}
    });
}());
