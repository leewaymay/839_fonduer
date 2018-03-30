document.CROSSMARK={VERIFICATION:"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ2ZXIiOiIyLjAuMTA2IiwiaWF0IjoxNTIyMDc5NDk2fQ.QM9hbjB8iplum1OUDzk8jNdRRu_spoatKeLy7j7Xq-I",ENDPOINT:"https://crossmark.crossref.org/dialog",SCRIPT_VERSION:"2.0.106",STYLESHEET_URL:"https://crossmark-cdn.crossref.org/widget/v2.0/style.css",LOGO_URL:"https://crossmark-cdn.crossref.org/images/logo-crossmark.svg"};
document.CROSSMARK.getDoiMetaTags=function(){var b=[],d=document.querySelectorAll("meta"),a;for(a in d)if(d.hasOwnProperty(a)){var c=d[a];if((c.name||"").toLowerCase().match(/^dc\.identifier/)){var e=(c.getAttribute("scheme")||"").toLowerCase();c.getAttribute("content")&&c.getAttribute("content").match(/(10\.\d+\/.*$)/)&&(""===e||"doi"===e)&&b.push(c)}}return b};
document.CROSSMARK.getDoi=function(){var b=document.CROSSMARK.getDoiMetaTags()[0];if(0===b.length)return null;var b=b?b.getAttribute("content").replace(/^(info:doi\/|doi:)/,""):null,d=b.match(/(10\.\d+\/.*$)/);null!==d&&(b=d[0]);return b};document.CROSSMARK.buildQueryString=function(b){var d=[],a;for(a in b)b.hasOwnProperty(a)&&d.push(encodeURIComponent(a)+"\x3d"+encodeURIComponent(b[a]));return"?"+d.join("\x26")};document.CROSSMARK.touchStarted=!1;document.CROSSMARK.touchArea=null;
document.CROSSMARK.tapEvent=function(b,d){!0!==b.gotEventListener&&(b.gotEventListener=!0,b.addEventListener("click",function(a){if(!(a.ctrlKey||a.shiftKey||a.metaKey||1!==a.which))return d(a)},!1),b.addEventListener("touchstart",function(a){1<a.touches.length?document.CROSSMARK.touchStarted=!1:(document.CROSSMARK.touchArea={x:a.touches[0].screenX,y:a.touches[0].screenY},document.CROSSMARK.touchStarted=b,a.stopPropagation())},!1),window.addEventListener("touchstart",function(a){document.CROSSMARK.touchStarted=
!1}),b.addEventListener("touchmove",function(a){if(1<a.touches.length)document.CROSSMARK.touchStarted=!1;else{var b=a.touches[0].screenY;500<Math.pow(document.CROSSMARK.touchArea.x-a.touches[0].screenX,2)+Math.pow(document.CROSSMARK.touchArea.y-b,2)&&(document.CROSSMARK.touchStarted=!1)}},!1),b.addEventListener("touchend",function(a){if(document.CROSSMARK.touchStarted)return document.CROSSMARK.touchStarted=!1,d(a);a.preventDefault()},!1))};
document.CROSSMARK.erase=function(){var b=document.querySelector(".crossmark-overlay");null!==b&&b.parentNode&&b.parentNode.removeChild(b)};
document.CROSSMARK.show=function(){document.CROSSMARK.erase();var b=/iPad|iPhone|iPod/.test(navigator.userAgent)&&!window.MSStream,d={doi:document.CROSSMARK.getDoi(),domain:window.location.hostname,uri_scheme:window.location.protocol,cm_version:document.CROSSMARK.SCRIPT_VERSION,verification:document.CROSSMARK.VERIFICATION},a=document.createElement("link");a.setAttribute("href",document.CROSSMARK.STYLESHEET_URL);a.setAttribute("type","text/css");a.setAttribute("rel","stylesheet");document.querySelector("head").appendChild(a);
var c=document.createElement("div");c.setAttribute("id","crossmark-widget");c.style.display="none";c.innerHTML='\x3cdiv class\x3d"crossmark-reset crossmark-overlay"\x3e\x3c/div\x3e\x3cdiv class\x3d"crossmark-reset crossmark-popup"\x3e\x3cdiv class\x3d"crossmark-reset crossmark-popup__offset"\x3e\x3cdiv class\x3d"crossmark-reset crossmark-popup__inner"\x3e\x3cdiv class\x3d"crossmark-reset crossmark-popup__header"\x3e\x3ca target\x3d"_blank" href\x3d"http://www.crossref.org/crossmark"\x3e\x3cimg class\x3d"crossmark-reset crossmark-popup__logo"\x3e\x3c/a\x3e\x3cbutton class\x3d"crossmark-reset crossmark-popup__btn-close"\x3e\x3c/button\x3e\x3c/div\x3e\x3cdiv class\x3d"crossmark-reset crossmark-popup__content-wrapper"\x3e\x3ciframe class\x3d"crossmark-reset crossmark-popup__content"\x3e\x3c/iframe\x3e\x3c/div\x3e\x3c/div\x3e\x3c/div\x3e\x3c/div\x3e';
var a=c.querySelector(".crossmark-overlay"),e=c.querySelector(".crossmark-popup"),g=c.querySelector(".crossmark-popup__offset"),f=c.querySelector(".crossmark-popup__inner"),h=c.querySelector(".crossmark-popup__logo"),k=c.querySelector(".crossmark-popup__content"),l=c.querySelector(".crossmark-popup__btn-close");k.setAttribute("src",document.CROSSMARK.ENDPOINT+document.CROSSMARK.buildQueryString(d));b&&g.classList.add("is-ios");h.setAttribute("src",document.CROSSMARK.LOGO_URL);document.body.appendChild(c);
[a,e,l].map(function(a){document.CROSSMARK.tapEvent(a,function(a){c.style.display="none";a.preventDefault();a.stopPropagation()})});document.CROSSMARK.tapEvent(f,function(a){a.stopPropagation()});c.style.display="block";b&&(f.style.top=window.scrollY+"px")};document.CROSSMARK.bind=function(b){[].slice.call(document.querySelectorAll("[data-target\x3dcrossmark]"),0).map(function(b){b.style.cursor="pointer";document.CROSSMARK.tapEvent(b,function(a){document.CROSSMARK.show();a.preventDefault();a.stopPropagation()})})};
document.addEventListener("DOMContentLoaded",document.CROSSMARK.bind);document.CROSSMARK.bind();