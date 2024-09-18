// Accéder aux classes via les variables globales
const { AutoTokenizer, AutoModelForCausalLM } = transformers;
const { ChatInterface } = gradio;

// Vérifier le support de WebAssembly
if (!('WebAssembly' in window)) {
  alert("Votre navigateur ne supporte pas WebAssembly. Veuillez utiliser un navigateur moderne comme Chrome ou Firefox.");
} else {
  main();
}

async function main() {
  // Afficher un message de chargement
  const gradioApp = document.getElementById('gradio-app');
  if (gradioApp) {
    gradioApp.innerText = 'Chargement du modèle...';
  }

  try {
    // Charger le modèle Gemini Nano
    const model = await AutoModelForCausalLM.from_pretrained('Xenova/gemini-nano', {
      progress_callback: (loaded: number, total: number) => {
        const percent = Math.round((loaded / total) * 100);
        if (gradioApp) {
          gradioApp.innerText = `Chargement du modèle... ${percent}%`;
        }
      }
    });
    const tokenizer = await AutoTokenizer.from_pretrained('Xenova/gemini-nano');

    // Préparer les réponses prédéfinies (enrichies)
    const predefinedAnswers: { [key: string]: string } = {
      "quelle est votre expérience": "J'ai travaillé en tant qu'ingénieur IA chez XYZ Company depuis 2020, développant des modèles de deep learning pour la reconnaissance d'images.",
      "quelles sont vos compétences": "Je maîtrise Python, le Machine Learning, le Deep Learning, le Traitement du Langage Naturel, et la Computer Vision.",
      "où avez-vous étudié": "J'ai obtenu un Master en Intelligence Artificielle à l'Université de Technologie en 2018.",
      "parlez-moi de votre projet de chatbot": "Mon projet de chatbot IA vise à créer un assistant virtuel capable de comprendre et de répondre aux questions des utilisateurs de manière naturelle.",
      "comment puis-je vous contacter": "Vous pouvez me contacter par email à ibrahim.mohammad@example.com.",
      // Ajoutez d'autres questions et réponses ici
    };

    // Charger les conversations sauvegardées
    let conversationHistory: Array<{ user: string, bot: string }> = JSON.parse(localStorage.getItem('conversationHistory') || '[]');

    // Définir la fonction de prédiction
    async function predict(inputText: string): Promise<string> {
      const lowerInput = inputText.toLowerCase().trim();

      // Vérifier si une réponse prédéfinie existe
      if (predefinedAnswers[lowerInput]) {
        const response = predefinedAnswers[lowerInput];
        saveConversation(inputText, response);
        return response;
      }

      // Sinon, utiliser le modèle pour générer une réponse
      const encoded = tokenizer.encode(inputText, { add_special_tokens: true });
      const inputIds = encoded.input_ids;

      const output = await model.generate(inputIds, {
        max_new_tokens: 50,
        do_sample: true,
        temperature: 0.7,
      });

      const decodedText = tokenizer.decode(output[0], { skip_special_tokens: true });

      // Sauvegarder la conversation
      saveConversation(inputText, decodedText);

      return decodedText;
    }

    // Fonction pour sauvegarder les conversations
    function saveConversation(userInput: string, botResponse: string) {
      conversationHistory.push({ user: userInput, bot: botResponse });
      localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
    }

    // Fonction pour effacer l'historique des conversations
    function clearHistory() {
      conversationHistory = [];
      localStorage.removeItem('conversationHistory');
      app.clear();
    }

    // Créer l'interface Gradio avec l'historique des conversations
    const app = new ChatInterface(
      predict,
      {
        description: "Posez une question à l'assistant virtuel.",
        title: "Assistant Virtuel",
        theme: "default",
      }
    );

    // Charger l'historique des conversations
    if (conversationHistory.length > 0) {
      app.chat_history = conversationHistory.map(conv => [conv.user, conv.bot]);
    }

    // Monter l'application Gradio
    app.launch("#gradio-app");

    // Ajouter des événements aux boutons
    const clearButton = document.getElementById('clear-history');
    if (clearButton) {
      clearButton.addEventListener('click', clearHistory);
    }
    const downloadButton = document.getElementById('download-cv');
    if (downloadButton) {
      downloadButton.addEventListener('click', () => {
        window.open('assets/pdf/ibrahim_mohammad_cv.pdf', '_blank');
      });
    }

  } catch (error) {
    console.error("Erreur lors du chargement du modèle :", error);
    if (gradioApp) {
      gradioApp.innerText = 'Une erreur est survenue lors du chargement du modèle.';
    }
  }
}
