  const texts = [
    "I'm an electrical engineer.",
    "I enjoy coding.",
    "Circuit boards make me excited.",
    "I make electrons behave."

  ];

  const typedText = document.getElementById("typed-text");


  const typingSpeed = 60;
  const erasingSpeed = 30;
  let displaySentence = 5000;

  let textIndex = 0;
  let char = 0;

  function type() {
    if (char < texts[textIndex].length) {
      typedText.textContent += texts[textIndex][char];
      char++;
      setTimeout(type, typingSpeed); // schedule next character
    }
    else {
      displaySentence = 800 * texts[textIndex].split(" ").length  // more words = more time
      setTimeout(erase, displaySentence); // schedule next sentence
    }
  }


  function erase() {
    if (char > 0) {
      typedText.textContent = texts[textIndex].substring(0, char - 1);
      char--;
      setTimeout(erase, erasingSpeed)
    }
    else {
      textIndex = (textIndex + 1) % texts.length; // loop
      setTimeout(type, typingSpeed); // start next sentence
    }
  }


  document.addEventListener("DOMContentLoaded", type);
