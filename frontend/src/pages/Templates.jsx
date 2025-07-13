import React from "react";
import "./Templates.css";

const templateList = [
  {
    name: "Affidavit Template (Sample)",
    link: "https://www.legaltemplates.net/wp-content/uploads/2022/01/Affidavit-Template.pdf",
  },
  {
    name: "Rental Agreement (IndiaGov)",
    link: "https://www.india.gov.in/sites/upload_files/npi/files/sample_rental_agreement.pdf",
  },
  {
    name: "Power of Attorney (Sample)",
    link: "https://www.legalzoom.com/download/pdf/general-power-of-attorney.pdf",
  },
];

const Templates = () => {
  return (
    <div className="templates-page">
      <h2>📑 Legal Templates</h2>
      <ul>
        {templateList.map((t, i) => (
          <li key={i}>
            <a href={t.link} target="_blank" rel="noopener noreferrer">
              {t.name}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Templates;
