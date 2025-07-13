import React from 'react';
import './Glossary.css';

const glossaryData = {
  A: [
    { term: "Affidavit", link: "https://en.wikipedia.org/wiki/Affidavit" },
    { term: "Arbitration", link: "https://en.wikipedia.org/wiki/Arbitration" },
    { term: "Article 21", link: "https://en.wikipedia.org/wiki/Article_21_of_the_Constitution_of_India" },
  ],
  B: [
    { term: "Bail", link: "https://en.wikipedia.org/wiki/Bail" },
    { term: "Bench Trial", link: "https://en.wikipedia.org/wiki/Bench_trial" },
  ],
  C: [
    { term: "Constitutional Amendment", link: "https://en.wikipedia.org/wiki/Constitution_of_India#Amendments" },
    { term: "Contempt of Court", link: "https://en.wikipedia.org/wiki/Contempt_of_court" },
    { term: "Criminal Jurisdiction", link: "https://en.wikipedia.org/wiki/Jurisdiction#Criminal_justice" },
  ],
  D: [
    { term: "Decree", link: "https://en.wikipedia.org/wiki/Decree" },
    { term: "Digital Signature", link: "https://en.wikipedia.org/wiki/Digital_signature" },
  ],
  E: [
    { term: "Extradition", link: "https://en.wikipedia.org/wiki/Extradition" },
  ],
  F: [
    { term: "Fundamental Rights", link: "https://en.wikipedia.org/wiki/Fundamental_rights_in_India" },
  ],
  I: [
    { term: "Indian Penal Code", link: "https://en.wikipedia.org/wiki/Indian_Penal_Code" },
    { term: "Indian Evidence Act", link: "https://en.wikipedia.org/wiki/Indian_Evidence_Act" },
    { term: "Impeachment", link: "https://en.wikipedia.org/wiki/Impeachment_in_India" },
  ],
  J: [
    { term: "Jurisdiction", link: "https://en.wikipedia.org/wiki/Jurisdiction" },
  ],
  L: [
    { term: "Lok Adalat", link: "https://en.wikipedia.org/wiki/Lok_Adalat" },
  ],
  O: [
    { term: "Oath", link: "https://en.wikipedia.org/wiki/Oath#Oaths_and_swearing-in" },
    { term: "Order", link: "https://en.wikipedia.org/wiki/Order_(law)" },
  ],
  P: [
    { term: "Plea Bargain", link: "https://en.wikipedia.org/wiki/Plea_bargaining" },
    { term: "Plaintiff", link: "https://en.wikipedia.org/wiki/Plaintiff" },
    { term: "Presumption of Innocence", link: "https://en.wikipedia.org/wiki/Presumption_of_innocence" },
  ],
  Q: [
    { term: "Quash", link: "https://en.wikipedia.org/wiki/Quash_(law)" },
  ],
  R: [
    { term: "Riot", link: "https://en.wikipedia.org/wiki/Riot" },
  ],
  S: [
    { term: "Summary Judgment", link: "https://en.wikipedia.org/wiki/Summary_judgment" },
    { term: "Statutory Law", link: "https://en.wikipedia.org/wiki/Statutory_law" },
  ],
  T: [
    { term: "Tribunal", link: "https://en.wikipedia.org/wiki/Tribunal" },
    { term: "Trust", link: "https://en.wikipedia.org/wiki/Trust_law" },
  ],
  W: [
    { term: "Warrant", link: "https://en.wikipedia.org/wiki/Warrant_(law)" },
    { term: "Writ", link: "https://en.wikipedia.org/wiki/Writs_in_India" },
  ],
};

const Glossary = () => {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

  return (
    <div className="glossary-container">
      <h1>Glossary</h1>

      <div className="alphabet-nav">
        {alphabet.map((letter) => (
          <a key={letter} href={`#${letter}`}>
            {letter}
          </a>
        ))}
      </div>

      <div className="glossary-content">
        {alphabet.map((letter) => (
          <div key={letter} id={letter} className="glossary-section">
            <h2>{letter}</h2>
            <ul>
              {(glossaryData[letter] || []).map(({ term, link }) => (
                <li key={term}>
                  <a href={link} target="_blank" rel="noopener noreferrer">
                    {term}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Glossary;
