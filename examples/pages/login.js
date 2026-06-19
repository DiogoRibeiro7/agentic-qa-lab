// Kept external (not inline) so the page's serialized DOM does not contain the
// success marker text until it is actually rendered — otherwise a substring
// match on the DOM snapshot would "see" the literal in the script source.
document.getElementById("submit").addEventListener("click", function () {
  var user = document.getElementById("username").value.trim();
  var pass = document.getElementById("password").value.trim();
  var result = document.getElementById("result");
  if (user && pass) {
    result.textContent = "Welcome, " + user + "!";
    result.setAttribute("data-state", "success");
  } else {
    result.textContent = "Please fill in both fields.";
    result.setAttribute("data-state", "error");
  }
});
