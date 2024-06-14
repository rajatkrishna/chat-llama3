var socket = io();

function Message(name, img, side, msg) {
	return `
	      <div class="flex gap-2 ${side === "left" ? "mb-2" : "justify-end"}">
		<div class="grid ${side === "right" ? "mb-2" : ""} mx-4 max-w-prose">
		    <div class="${side === "right" ? "text-right" : ""} text-gray-900 text-sm font-semibold leading-snug pb-1">${name}</div>
		    <div class="px-3.5 py-2 shadow-md ${side === "right" ? "bg-stone-300 rounded-tr-none" : "bg-slate-400 rounded-tl-none"} rounded-lg justify-start items-center gap-3 inline-flex">

		      <span class="whitespace-pre-line text-gray-900 text-sm font-normal leading-snug">${msg}</span>

		    </div>
		    <div class="${side === "left" ? "justify-end" : "justify-start"} items-center inline-flex mb-2.5">
			<h6 class="text-gray-500 text-xs font-normal leading-4 py-1">${formatDate(new Date())}</h6>
		    </div>
		</div>
		${side === "right" ? `<img src='${img}' class='rounded-full w-10 h-11'>` : ""}
	      </div>
	    `;
}

function formatDate(date) {
    const h = "0" + date.getHours();
    const m = "0" + date.getMinutes();

    return `${h.slice(-2)}:${m.slice(-2)}`;
}

function showResponse(msg) {
    const msgHTML = Message("Assistant", "static/img/logo-color.svg", "left", msg.answer);

    const messages = document.querySelector("#message-holder");
    messages.insertAdjacentHTML("beforeend", msgHTML);
    window.scrollTo(0, document.body.scrollHeight)
}

function showToksPSecond(tps) {
    $('#tps').text(tps.toFixed(4));
}

function handleUserMsg(e) {
    e.preventDefault()
    let msg = $('#chat').val()
    if (msg === undefined || msg == "") {
	return
    }

    socket.emit('msg_receive', {
	    message: msg
    })

    const msgHTML = Message("You", "static/images/usericon.jpeg", "right", msg, [])

    const messages = document.querySelector("#message-holder");
    messages.insertAdjacentHTML("beforeend", msgHTML);
    window.scrollTo(0, document.body.scrollHeight)
    $('#chat').val('').focus()
}

$('form').on('submit', handleUserMsg)
socket.on('msg_response', showResponse)
socket.on('tps_measure', showToksPSecond)
